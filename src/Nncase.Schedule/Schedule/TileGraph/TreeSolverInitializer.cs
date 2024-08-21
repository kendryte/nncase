// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Google.OrTools.ConstraintSolver;
using NetFabric.Hyperlinq;
using Nncase.IR.Affine;
using QuikGraph;
using QuikGraph.Graphviz;

namespace Nncase.Schedule.TileGraph;

public sealed class GraphSolverInitializer : GraphSolverBase, ITileGraphVisitor<GraphSolverInitializer.Context, GraphSolverInitializer.InitResult>
{
    public GraphSolverInitializer(int totalLevel, Solver solver, Dictionary<OpNode, OpNodeInfo> primitiveBufferInfo, Dictionary<TileGraph, TileNodeInfo> levelBufferInfos, Dictionary<ITileableNode, DomainInfo> domainDimInfos, ITargetOptions targetOptions)
        : base(solver, primitiveBufferInfo, levelBufferInfos, domainDimInfos, targetOptions)
    {
        TotalLevel = totalLevel;
    }

    public int TimeStamp { get; private set; }

    public int TotalLevel { get; }

    public static ArgumentsInfo Init(TileGraph tree, int totalLevel, ITargetOptions options, out Solver solver, out Dictionary<OpNode, OpNodeInfo> opNodeMemo, out Dictionary<TileGraph, TileNodeInfo> tileNodeMemo, out Dictionary<ITileableNode, DomainInfo> tileableNodeMemo)
    {
        solver = new Solver("GraphSolver");
        opNodeMemo = new Dictionary<OpNode, OpNodeInfo>();
        tileNodeMemo = new Dictionary<TileGraph, TileNodeInfo>();
        tileableNodeMemo = new Dictionary<ITileableNode, DomainInfo>();
        var initializer = new GraphSolverInitializer(totalLevel, solver, opNodeMemo, tileNodeMemo, tileableNodeMemo, options);
        var initResult = initializer.Visit(tree, Context.Default);
        var (graph, defuseMap) = LinkGraphs(tree.RootParent(), initResult, true);
        var (inputs, outputs) = GetGetGraphInputsOuputs(graph);
        return new(inputs, outputs, defuseMap);
    }

    /// <summary>
    /// source id => sink id.
    /// </summary>
    public static Dictionary<BufferIdentity, BufferIdentity> GetBufferDefUseMap(TileGraph graph, BufferResult[] bufferResults)
    {
        while (graph.Parent is TileGraph parent)
        {
            graph = parent;
        }

        var map = new Dictionary<BufferIdentity, BufferIdentity>();
        for (int i = 0; i < bufferResults.Length; i++)
        {
            var sourceId = bufferResults[i].Bid;
            if (graph.TryGetOutEdges(sourceId.Node, out var outEdges))
            {
                foreach (var outEdge in outEdges)
                {
                    foreach (var target in bufferResults.Where(r => r.Bid.Node == outEdge.Target && r.Bid.Index == outEdge.Index))
                    {
                        if (!map.ContainsKey(sourceId))
                        {
                            map.Add(sourceId, target.Bid);
                        }
                    }
                }
            }
        }

        return map;
    }

    public static void Dump(AdjacencyGraph<BufferIdentity, Edge<BufferIdentity>> graph, string name)
    {
        using (var file = Diagnostics.DumpScope.Current.OpenFile($"{name}.dot"))
        {
            using (var writer = new StreamWriter(file))
            {
                writer.WriteLine(graph.ToGraphviz(init =>
                {
                    init.FormatVertex += (_, args) => args.VertexFormat.Label = args.Vertex.ToString();
                }));
            }
        }
    }

    public static (HashSet<BufferIdentity> Inputs, HashSet<BufferIdentity> Outputs) GetGetGraphInputsOuputs(AdjacencyGraph<BufferIdentity, Edge<BufferIdentity>> g)
    {
        var sources = new HashSet<BufferIdentity>();
        var targets = new HashSet<BufferIdentity>();
        foreach (var item in g.Edges)
        {
            sources.Add(item.Source);
            targets.Add(item.Target);
        }

        var inputs = new HashSet<BufferIdentity>(sources.Except(targets));
        var outputs = new HashSet<BufferIdentity>(targets.Except(sources));
        return (inputs, outputs);
    }

    public static (AdjacencyGraph<BufferIdentity, Edge<BufferIdentity>> Graph, Dictionary<BufferIdentity, BufferIdentity> DefUseMap) LinkGraphs(TileGraph tileGraph, InitResult result, bool gatherCacheBuffers)
    {
        var graphIOs = new Dictionary<int, (HashSet<BufferIdentity> Inputs, HashSet<BufferIdentity> Outputs)>();

        for (int i = 0; i < result.Graphs.Length; i++)
        {
            graphIOs[i] = GetGetGraphInputsOuputs(result.Graphs[i]);
        }

        var graph = new AdjacencyGraph<BufferIdentity, Edge<BufferIdentity>>();
        for (int i = 0; i < result.Graphs.Length; i++)
        {
            var subGraph = result.Graphs[i];
            foreach (var edge in subGraph.Edges)
            {
                graph.AddVerticesAndEdge(edge);
            }
        }

        var defMaps = new Dictionary<BufferIdentity, BufferIdentity>(result.DefUseMap);

        for (int i = 0; i < result.Graphs.Length; i++)
        {
            for (int j = i + 1; j < result.Graphs.Length; j++)
            {
                // connect the producer outputs and consumer inputs.
                var producers = graphIOs[i].Outputs;
                var consumers = graphIOs[j].Inputs;
                foreach (var (producer, consumer) in new[] { producers, consumers }.CartesianProduct().Select(x => (x.First(), x.Skip(1).First())))
                {
                    if (tileGraph.ContainsEdge(new OpEdge(producer.Node, consumer.Node, consumer.Index)))
                    {
                        graph.AddEdge(new(producer, consumer));
                        if (gatherCacheBuffers && !defMaps.ContainsKey(producer))
                        {
                            defMaps.TryAdd(producer, consumer);
                        }
                    }
                }
            }
        }

        return (graph, defMaps);
    }

    public InitResult Visit(TileGraph value, Context context)
    {
        var (pid, pvars) = context;
        var dimsMap = GetDimsMap(value);
        if (!pvars.Any())
        {
            dimsMap.Clear();
        }

        var tileVars = Enumerable.Range(0, value.DomainRelation.Map.Results.Length).Select(n => Solver.MakeIntVar(1, int.MaxValue, $"{n}_L{value.Level}")).ToArray();
        var forwardExtents = tileVars.Cast<IntExpr>().ToArray();
        if (!TileableNodeMemo.TryGetValue(value, out var dimInfo))
        {
            foreach (var (k, v) in dimsMap)
            {
                forwardExtents[k] *= pvars[v];
            }

            TileableNodeMemo.Add(value, new(tileVars, forwardExtents, dimsMap));
        }

        InitResult childResult;
        {
            var childContext = context with { ParentOpId = value.OpId, ForwardExtents = forwardExtents };

            var results = new List<BufferResult>();
            var graphs = new List<AdjacencyGraph<BufferIdentity, Edge<BufferIdentity>>>();
            var names = new List<Dictionary<int, int>>();
            var extents = new List<IntExpr[]>();
            var childDefUseMap = new Dictionary<BufferIdentity, BufferIdentity>();
            foreach (var (i, child) in Enumerable.Range(0, value.ClustersCount).Zip(value.Clusters.OfType<TileGraph>()))
            {
                var res = Visit(child, childContext);
                results.AddRange(res.BufferResults);
                graphs.AddRange(res.Graphs);
                extents.AddRange(res.BackWardExtents);
                names.AddRange(res.DimsMaps);
                foreach (var (k, v) in res.DefUseMap)
                {
                    childDefUseMap.Add(k, v);
                }
            }

            childResult = new(results.ToArray(), graphs.ToArray(), childDefUseMap, names.ToArray(), extents.ToArray());
        }

        var backWardExtents = GetBackWardExtents(tileVars, childResult.DimsMaps, childResult.BackWardExtents);

        var defUseMap = GetBufferDefUseMap(value, childResult.BufferResults);
        var bufferResults = new List<BufferResult>();

        // each tile node have buffer place vars.
        if (!TileNodeMemo.TryGetValue(value, out var info))
        {
            var bufferInfoMap = new Dictionary<BufferIdentity, TileNodeBufferInfo>();
            for (int i = 0; i < childResult.BufferResults.Length; i++)
            {
                var result = childResult.BufferResults[i];
                BufferIdentity currentId;
                AffineMap currentAccessMap = result.AccessMap;
                Tuple<int, int> currentLifeness = result.Lifeness;
                if (defUseMap.TryGetValue(result.Bid, out currentId!))
                {
                    var sinkIndex = Array.FindIndex(childResult.BufferResults, r => r.Bid == currentId);
                    currentAccessMap = childResult.BufferResults[sinkIndex].AccessMap;
                    currentLifeness = new(Math.Min(result.Lifeness.Item1, childResult.BufferResults[sinkIndex].Lifeness.Item1), Math.Max(result.Lifeness.Item2, childResult.BufferResults[sinkIndex].Lifeness.Item2));
                }
                else
                {
                    currentId = result.Bid;
                }

                if (!bufferInfoMap.TryGetValue(currentId, out var bufferInfo))
                {
                    bufferInfoMap.Add(currentId, GetBufferInfo(value, currentId, currentAccessMap, currentLifeness, backWardExtents));
                    bufferResults.Add(new(currentId, currentLifeness, value.DomainRelation.Map * currentAccessMap));
                }
            }

            TileNodeMemo.Add(value, new(backWardExtents, defUseMap, bufferInfoMap));
        }

        // link the graphs
        var (graph, retDefUseMap) = LinkGraphs(value, childResult, value.Level == TotalLevel);

        return new(bufferResults.ToArray(), new[] { graph }, retDefUseMap, new[] { dimsMap }, new[] { backWardExtents[0] });
    }

    public InitResult Visit(OpNode value, Context context)
    {
        var (pid, pvars) = context;
        var dimsMap = GetDimsMap(value);
        var tileVars = Enumerable.Range(0, value.DomainBounds.Length).Select(n => Solver.MakeIntVar(1, long.MaxValue, $"d{n}_L{value.Level}")).ToArray();

        // CompilerServices.GetOpMicroKernelInfo(value.Op, value.AccessMaps[0].Domains.AsValueEnumerable().Select(i => i.Offset).ToArray(), value.AccessMaps.ToArray(), value.BufferShapes, TargetOptions);
        var kernelInfo = new MicroKernelInfo(tileVars.Select(i => 1).ToArray(), tileVars.Select((_, i) => new ValueRange<int>(0, value.DomainBounds[i])).ToArray(), 1, 1);

        for (int i = 0; i < tileVars.Length; i++)
        {
            tileVars[i].SetRange(kernelInfo.Multiplier[i].Min, kernelInfo.Multiplier[i].Max);
        }

        var primtiveMap = AffineMap.FromCallable((doms, syms) => doms.Select(i => new AffineRange(i.Offset, kernelInfo.Primitives[i.Extent.Position] * i.Extent)).ToArray(), value.DomainBounds.Length);
        var accessMaps = new AffineMap[value.BufferShapes.Length];

        // cache the primitive buffer shape and sizes.
        if (!OpNodeMemo.TryGetValue(value, out var info))
        {
            var shapes = new IntExpr[value.BufferShapes.Length][];
            var sizes = new IntExpr[value.BufferShapes.Length];
            for (int a = 0; a < value.BufferShapes.Length; a++)
            {
                shapes[a] = new IntExpr[value.BufferShapes[a].Length];
                sizes[a] = Solver.MakeIntConst(value.Grid.Buffers[a].CheckedDataType.SizeInBytes);
                var extentVars = tileVars;
                var converter = new AffineExprToIntExprConverter(Solver, extentVars);
                accessMaps[a] = primtiveMap * value.Grid.AccessMaps[a];
                for (int i = 0; i < shapes[a].Length; i++)
                {
                    shapes[a][i] = converter.Visit(accessMaps[a].Results[i].Extent);
                    sizes[a] *= shapes[a][i];
                }
            }

            OpNodeMemo.Add(value, new(accessMaps, shapes, sizes));
        }

        if (!TileableNodeMemo.TryGetValue(value, out var dimInfo))
        {
            var forwardExtents = tileVars.Cast<IntExpr>().ToArray();
            foreach (var (i, j) in dimsMap)
            {
                forwardExtents[i] *= pvars[j];
            }

            for (int i = 0; i < tileVars.Length; i++)
            {
                forwardExtents[i] *= kernelInfo.Primitives[i];
            }

            TileableNodeMemo.Add(value, new(tileVars, forwardExtents, dimsMap));
        }

        // perpare return infos.
        var bufferResults = new BufferResult[value.ReadAccesses.Length + 1];
        var graph = new AdjacencyGraph<BufferIdentity, Edge<BufferIdentity>>();
        BufferIdentity obid = new(value, value.ReadAccesses.Length);
        graph.AddVertex(obid);
        bufferResults[value.ReadAccesses.Length] = new(obid, new(TimeStamp, TimeStamp + 1), value.DomainRelation.Map * accessMaps[^1]);

        for (int i = 0; i < value.ReadAccesses.Length; i++)
        {
            BufferIdentity bid = new(value, i);
            graph.AddVertex(bid);
            graph.AddEdge(new(bid, obid));
            bufferResults[i] = new(bid, new(TimeStamp, TimeStamp + 1), value.DomainRelation.Map * accessMaps[i]);
        }

        TimeStamp += 2;

        // todo backward extents should times primtives.
        return new(bufferResults, new[] { graph }, new(), new[] { dimsMap }, new IntExpr[][] { tileVars.Cast<IntExpr>().ToArray() });
    }

    /// <summary>
    /// Get the backward accumulated domain extents, domain extents[i] means extents[0:i] is not accumulated, extents[i:] is accumulated.
    /// </summary>
    private IntExpr[][] GetBackWardExtents(IntVar[] tileVars, Dictionary<int, int>[] childDimsMaps, IntExpr[][] childBackWardExtents)
    {
        var backWardExtents = new IntExpr[tileVars.Length][];
        bool ProductExtent(IntExpr[] extents, int i)
        {
            bool find = false;
            for (int cid = 0; cid < childDimsMaps.Length; cid++)
            {
                var cmap = childDimsMaps[cid];
                var cextents = childBackWardExtents[cid];
                foreach (var (k, v) in cmap)
                {
                    if (i == v)
                    {
                        extents[v] = extents[v] is null ? cextents[k] : extents[v] * cextents[k];
                        return find;
                    }
                }
            }

            throw new InvalidOperationException("can't find the child tile var");
        }

        for (int d = 0; d < tileVars.Length; d++)
        {
            var extents = backWardExtents[d] = new IntExpr[tileVars.Length];
            for (int i = 0; i < d; i++)
            {
                ProductExtent(extents, i);
            }

            for (int i = d; i < tileVars.Length; i++)
            {
                extents[i] = tileVars[i];
                ProductExtent(extents, i);
            }
        }

        return backWardExtents;
    }

    private TileNodeBufferInfo GetBufferInfo(TileGraph tile, BufferIdentity bid, AffineMap accessMap, Tuple<int, int> lifeness, IntExpr[][] backWardExtents)
    {
        var domainDims = tile.DomainRelation.Map.Results.Length;
        var bufferPlaces = new IntVar[domainDims][];
        var bufferShapes = new IntExpr[domainDims][];
        var bufferWrites = new IntExpr[domainDims];
        var bufferSizes = new IntExpr[domainDims];
        var bufferSizeVars = new IntVar[domainDims];
        var bufferMasks = new LoopMask[domainDims];

        for (int i = 0; i < domainDims; i++)
        {
            // note we assume the inputs/outputs already stored at top level, so disable the top level store buffer placement.
            var subLevelPlace = bufferPlaces[i] = new IntVar[tile.Level == TotalLevel ? tile.Level - 1 : tile.Level];
            for (int sl = 0; sl < subLevelPlace.Length; sl++)
            {
                subLevelPlace[sl] = Solver.MakeBoolVar($"p[cl{tile.Level}, op{tile.OpId}, b{bid.Index}, ci{i}, {sl}]");
            }

            var subDomainShapes = bufferShapes[i] = new IntExpr[accessMap.Results.Length];
            var converter = new AffineExprToIntExprConverter(Solver, backWardExtents[i]);
            for (int j = 0; j < accessMap.Results.Length; j++)
            {
                subDomainShapes[j] = converter.Visit(accessMap.Results[j].Extent);
            }

            bufferSizes[i] = subDomainShapes.Aggregate((IntExpr)Solver.MakeIntConst(bid.Node.Grid.Buffers[bid.Index].CheckedDataType.SizeInBytes), Solver.MakeProd);
            bufferSizeVars[i] = Solver.MakeIntVar(1, int.MaxValue, $"size[cl{tile.Level}, op{tile.OpId}, b{bid.Index}, ci{i}]");
            Solver.Add(Solver.MakeEquality(bufferSizeVars[i], bufferSizes[i]));

            var mask = 0U;
            var sizeStr = bufferSizes[i].ToString();
            for (int j = 0; j < domainDims; j++)
            {
                if (sizeStr.Contains(TileableNodeMemo[tile].TileVars[i].Name(), StringComparison.CurrentCulture))
                {
                    mask |= 1U << j;
                }
            }

            bufferMasks[i] = new(mask);

            // note update writes in second visitor.
        }

        var bufferInfo = new TileNodeBufferInfo(lifeness, accessMap, bufferPlaces, bufferShapes, bufferWrites, bufferSizeVars, bufferSizes, bufferMasks);
        return bufferInfo;
    }

    /// <summary>
    /// each buffer with each access Maps, note the access map domain is this node's domain. extents also mapping to current node's domain.
    /// </summary>
    /// <param name="BufferResults">buffer info.</param>
    /// <param name="Graphs"> sub computation graph. </param>
    /// <param name="DefUseMap">the defuse map is used to record cache buffer in the top memory level. </param>
    /// <param name="DimsMaps">dims map.</param>
    /// <param name="BackWardExtents"> backward extents for cout the buffer size. </param>
    public sealed record InitResult(BufferResult[] BufferResults, AdjacencyGraph<BufferIdentity, Edge<BufferIdentity>>[] Graphs, Dictionary<BufferIdentity, BufferIdentity> DefUseMap, Dictionary<int, int>[] DimsMaps, IntExpr[][] BackWardExtents)
    {
    }

    /// <summary>
    /// buffer init result.
    /// </summary>
    /// <param name="Bid">buffer id.</param>
    /// <param name="Lifeness">buffer's lifetime.</param>
    /// <param name="AccessMap">access buffer relation from current node's domain, e.g. node.DomainRelation * buffer.AccessMap.</param>
    public sealed record BufferResult(BufferIdentity Bid, Tuple<int, int> Lifeness, AffineMap AccessMap)
    {
    }

    public sealed record Context(int ParentOpId, IReadOnlyList<IntExpr> ForwardExtents)
    {
        public static Context Default => new(-1, Array.Empty<IntVar>());
    }
}
