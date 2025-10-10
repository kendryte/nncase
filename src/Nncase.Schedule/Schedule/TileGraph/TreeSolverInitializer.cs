// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Google.OrTools.ConstraintSolver;
using NetFabric.Hyperlinq;
using Nncase.IR.Affine;
using QuikGraph;
using QuikGraph.Graphviz;
using static Nncase.TIR.TIRExtensions;

namespace Nncase.Schedule.TileGraph;

public sealed class TreeSolverInitializer : TreeSolverBase<IntExpr>, ITreeNodeVisitor<TreeSolverInitializer.Context, TreeSolverInitializer.InitResult>
{
    public TreeSolverInitializer(Dictionary<TieredTileGraph, BufferGraph> bufferGraphMemo, int levelCount, Solver solver, Dictionary<OpNode, OpNodeInfo<IntExpr>> primitiveBufferInfo, Dictionary<TileNode, TileNodeInfo<IntExpr>> levelBufferInfos, Dictionary<ITileable, DomainInfo<IntExpr>> domainDimInfos, INTTTargetOptions targetOptions)
        : base(solver, primitiveBufferInfo, levelBufferInfos, domainDimInfos, targetOptions)
    {
        BufferGraphMemo = bufferGraphMemo;
        TopLevel = levelCount;
    }

    public int TimeStamp { get; private set; }

    public IReadOnlyDictionary<TieredTileGraph, BufferGraph> BufferGraphMemo { get; }

    public int TopLevel { get; }

    public static void Init(TileNode tree, Dictionary<TieredTileGraph, BufferGraph> bufferGraphMemo, int levelCount, INTTTargetOptions options, out Solver solver, out Dictionary<OpNode, OpNodeInfo<IntExpr>> opNodeMemo, out Dictionary<TileNode, TileNodeInfo<IntExpr>> tileNodeMemo, out Dictionary<ITileable, DomainInfo<IntExpr>> tileableNodeMemo)
    {
        solver = new Solver("GraphSolver");
        opNodeMemo = new Dictionary<OpNode, OpNodeInfo<IntExpr>>();
        tileNodeMemo = new Dictionary<TileNode, TileNodeInfo<IntExpr>>();
        tileableNodeMemo = new Dictionary<ITileable, DomainInfo<IntExpr>>();
        var initializer = new TreeSolverInitializer(bufferGraphMemo, levelCount, solver, opNodeMemo, tileNodeMemo, tileableNodeMemo, options);
        initializer.Visit(tree, Context.Default);
    }

    public InitResult Visit(TileNode value, Context context)
    {
        var (pid, pvars, ptrips) = context;
        var dimsMap = GetDimsMap(value);
        if (!pvars.Any())
        {
            dimsMap.Clear();
        }

        var tileVars = Enumerable.Range(0, value.DomainRelation.Map.Results.Length).Select(n => Solver.MakeIntVar(1, int.MaxValue, $"op{value.OpId}_d{n}_L{value.Level}")).ToArray();
        var forwardExtents = tileVars.Cast<IntExpr>().ToArray();
        if (!TileableNodeMemo.TryGetValue(value, out var dimInfo))
        {
            foreach (var (k, v) in dimsMap)
            {
                forwardExtents[k] *= pvars[v];
            }

            TileableNodeMemo.Add(value, new(tileVars, forwardExtents, dimsMap));
        }

        var tripCounts = new IntExpr[tileVars.Length + 1];
        if (pvars.Any())
        {
            tripCounts[0] = ptrips;
        }
        else
        {
            tripCounts[0] = Solver.MakeIntConst(1);
        }

        for (int i = 0; i < tileVars.Length; i++)
        {
            tripCounts[1 + i] = tripCounts[i] * tileVars[i];
        }

        InitResult childResult;
        {
            var childContext = context with { ParentOpId = value.OpId, ForwardExtents = forwardExtents, TripCounts = tripCounts[^1] };

            var results = new List<BufferResult>();
            var names = new List<Dictionary<int, int>>();
            var extents = new List<IntExpr[]>();
            var childDefUseMap = new BiDictionary<BufferIdentity, BufferIdentity>();
            foreach (var child in value.Children)
            {
                var res = child.Accept(this, childContext);
                results.AddRange(res.BufferResults);
                extents.AddRange(res.BackWardExtents);
                names.AddRange(res.DimsMaps);
                foreach (var (k, v) in res.DefUseMap)
                {
                    childDefUseMap.Add(k, v);
                }
            }

            childResult = new(results.ToArray(), childDefUseMap, names.ToArray(), extents.ToArray());
        }

        var backWardExtents = GetBackWardExtents(tileVars, childResult.DimsMaps, childResult.BackWardExtents);

        // {def bid : use bid}
        var defUseMap = BufferGraphMemo[value.Wrapped].Edges.Where(e => e.Tag == BufferEdgeKind.Inter).ToBiDictionary(e => e.Source, e => e.Target);
        var bufferResults = new List<BufferResult>();

        // gather the min/max lifeness child.
        Tuple<int, int> nodeLifeness = new(int.MaxValue, int.MinValue);
        for (int i = 0; i < childResult.BufferResults.Length; i++)
        {
            var lifeness = childResult.BufferResults[i].Lifeness;
            nodeLifeness = new(Math.Min(nodeLifeness.Item1, lifeness.Item1), Math.Max(nodeLifeness.Item2, lifeness.Item2));
        }

        // each tile node have buffer place vars.
        if (!TileNodeMemo.TryGetValue(value, out var info))
        {
            var bufferInfoMap = new Dictionary<BufferIdentity, TileNodeBufferInfo<IntExpr>>();
            for (int i = 0; i < childResult.BufferResults.Length; i++)
            {
                var result = childResult.BufferResults[i];
                var curId = result.Bid;
                if (defUseMap.ContainsValue(curId))
                {
                    continue;
                }

                AffineMap currentAccessMap = result.AccessMap;
                Tuple<int, int> currentLifeness = result.Lifeness;
                if (defUseMap.TryGetByKey(curId, out var sinkBIds))
                {
                    foreach (var sinkBid in sinkBIds)
                    {
                        if (Array.FindIndex(childResult.BufferResults, r => r.Bid == sinkBid) is var sinkIndex && sinkIndex != -1)
                        {
                            currentAccessMap = childResult.BufferResults[sinkIndex].AccessMap;
                            currentLifeness = new(Math.Min(currentLifeness.Item1, childResult.BufferResults[sinkIndex].Lifeness.Item1), Math.Max(currentLifeness.Item2, childResult.BufferResults[sinkIndex].Lifeness.Item2));
                        }
                    }
                }

                if (!bufferInfoMap.TryGetValue(curId, out var bufferInfo))
                {
                    bufferInfo = GetBufferInfo(value, curId, currentAccessMap, nodeLifeness, currentLifeness, tileVars, forwardExtents, backWardExtents, result.ElemSize);
                    bufferInfoMap.Add(curId, bufferInfo);
                    bufferResults.Add(new(curId, currentLifeness, value.DomainRelation.Map * currentAccessMap, result.ElemSize));
                }
            }

            TileNodeMemo.Add(value, new(tripCounts, backWardExtents, defUseMap, bufferInfoMap));
        }

        return new(bufferResults.ToArray(), defUseMap, new[] { dimsMap }, new[] { backWardExtents[0] });
    }

    public InitResult Visit(OpNode value, Context context)
    {
        var (pid, pvars, ptrips) = context;
        var dimsMap = GetDimsMap(value);
        var tileVars = Enumerable.Range(0, value.DomainBounds.Length).Select(n => Solver.MakeIntVar(1, long.MaxValue, $"op{value.OpId}_d{n}")).ToArray();

        var kernelInfo = value.GetKernelInfo(TargetOptions);

        for (int i = 0; i < tileVars.Length; i++)
        {
            tileVars[i].SetRange(kernelInfo.TileBounds[i].Min, kernelInfo.TileBounds[i].Max);
        }

        // cache the primitive buffer shape and sizes.
        var accessMaps = new AffineMap[value.BufferShapes.Length];
        var elemSizes = new IntExpr[value.BufferShapes.Length];
        if (!OpNodeMemo.TryGetValue(value, out var opInfo))
        {
            var shapes = new IntExpr[value.BufferShapes.Length][];
            var sizes = new IntExpr[value.BufferShapes.Length];
            for (int a = 0; a < value.BufferShapes.Length; a++)
            {
                shapes[a] = new IntExpr[value.BufferShapes[a].Length];
                elemSizes[a] = sizes[a] = Solver.MakeIntConst(value.GetBufferElemSize(a));
                accessMaps[a] = value.Grid.AccessMaps[a];
                var converter = new AffineExprToIntExprConverter(Solver, tileVars);
                var accessMap = value.DomainRelation.Map * value.Grid.AccessMaps[a];
                for (int i = 0; i < shapes[a].Length; i++)
                {
                    // note themory solution, the custom affine map is not suitable for compute buffer size.
                    if (accessMap.Results[i].Offset is AffineConstant c)
                    {
                        shapes[a][i] = converter.Visit(accessMap.Results[i].Offset) + converter.Visit(accessMap.Results[i].Extent);
                    }
                    else
                    {
                        shapes[a][i] = converter.Visit(accessMap.Results[i].Offset);
                    }

                    sizes[a] *= shapes[a][i];
                }
            }

            opInfo = new(accessMaps, shapes, sizes);
            OpNodeMemo.Add(value, opInfo);
        }

        if (!TileableNodeMemo.TryGetValue(value, out var dimInfo))
        {
            var forwardExtents = tileVars.Cast<IntExpr>().ToArray();
            foreach (var (i, j) in dimsMap)
            {
                forwardExtents[i] *= pvars[j];
            }

            TileableNodeMemo.Add(value, new(tileVars, forwardExtents, dimsMap));
        }

        // perpare return infos.
        var bufferResults = new BufferResult[value.ReadAccesses.Length + 1];
        BufferIdentity obid = new(value.Wrapped, value.ReadAccesses.Length);
        bufferResults[value.ReadAccesses.Length] = new(obid, new(TimeStamp, TimeStamp + 1), value.DomainRelation.Map * accessMaps[^1], elemSizes[value.ReadAccesses.Length]);

        for (int i = 0; i < value.ReadAccesses.Length; i++)
        {
            BufferIdentity bid = new(value.Wrapped, i);
            bufferResults[i] = new(bid, new(TimeStamp, TimeStamp + 1), value.DomainRelation.Map * accessMaps[i], elemSizes[i]);
        }

        TimeStamp += 2;

        // todo backward extents should times primtives.
        return new(bufferResults, new(), new[] { dimsMap }, new IntExpr[][] { tileVars.Cast<IntExpr>().ToArray() });
    }

    /// <summary>
    /// Get the backward accumulated domain extents.
    /// backWardExtents[i] contains a extents[domain rank], note the extents[0:i] is not accumulated, extents[i:] is accumulated.
    /// for example. backWardExtents[2] contains extents[3], this extents[0],extents[1] is not accumulated, extents[2] is accumulated.
    /// so backWardExtents[0] means extents[0:domain rank] is accumulated.
    /// </summary>
    private IntExpr[][] GetBackWardExtents(IntVar[] tileVars, Dictionary<int, int>[] childDimsMaps, IntExpr[][] childBackWardExtents)
    {
        var backWardExtents = new IntExpr[tileVars.Length + 1][];
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

        for (int i = 0; i < tileVars.Length + 1; i++)
        {
            var extents = backWardExtents[i] = new IntExpr[tileVars.Length];

            // [0:i] is not accumulated.
            for (int j = 0; j < i; j++)
            {
                ProductExtent(extents, j);
            }

            // [i:domain] is accumulated
            for (int j = i; j < tileVars.Length; j++)
            {
                extents[j] = tileVars[j];
                ProductExtent(extents, j);
            }
        }

        return backWardExtents;
    }

    private TileNodeBufferInfo<IntExpr> GetBufferInfo(TileNode tileNode, BufferIdentity bid, AffineMap accessMap, Tuple<int, int> nodeLiveness, Tuple<int, int> currentLiveness, IntExpr[] tileVars, IntExpr[] forwardExtents, IntExpr[][] backWardExtents, IntExpr elemSize)
    {
        var rank = tileNode.DomainRelation.Map.Results.Length;
        var fullPos = rank + 1;
        var levelCount = tileNode.Level + 1;
        var bufferPlaces = Enumerable.Range(0, fullPos).Select(i => Array.Empty<IntExpr>()).ToArray();
        var bufferShapes = Enumerable.Range(0, fullPos).Select(i => Array.Empty<IntExpr>()).ToArray();
        var bufferSizes = new IntExpr[fullPos];
        var bufferTrips = new IntExpr[fullPos];
        var bufferLiveness = new Tuple<int, int>[fullPos];
        bufferLiveness[^1] = currentLiveness;
        for (int i = 0; i < fullPos - 1; i++)
        {
            bufferLiveness[i] = nodeLiveness;
        }

        LoopMask bufferMask = new(0);

        var resultStr = accessMap.ToString().Split("->")[1];
        for (int pos = 0; pos < fullPos; pos++)
        {
            var subLevelPlace = bufferPlaces[pos] = new IntVar[levelCount];
            for (int sl = 0; sl < subLevelPlace.Length; sl++)
            {
                subLevelPlace[sl] = Solver.MakeBoolVar($"p[cl{tileNode.Level}, op{bid.Node.OpId}, b{bid.Index}, ci{pos}, sl{sl}]");
            }

            var subDomainShapes = bufferShapes[pos] = new IntExpr[accessMap.Results.Length];
            var converter = new AffineExprToIntExprConverter(Solver, backWardExtents[pos]);
            for (int j = 0; j < accessMap.Results.Length; j++)
            {
                if (accessMap.Results[j].Offset is AffineConstant c)
                {
                    subDomainShapes[j] = converter.Visit(accessMap.Results[j].Offset) + converter.Visit(accessMap.Results[j].Extent);
                }
                else
                {
                    subDomainShapes[j] = converter.Visit(accessMap.Results[j].Offset);
                }
            }

            bufferSizes[pos] = subDomainShapes.Aggregate(elemSize, Solver.MakeProd);
            bufferSizes[pos].SetName($"size[cl{tileNode.Level}, op{bid.Node.OpId}, b{bid.Index}, ci{pos}]");

            var loop = pos - 1;
            if (loop < 0)
            {
                bufferTrips[pos] = Solver.MakeIntConst(1);
            }
            else
            {
                // todo use isl for detect reuse dims.
                var accessed = resultStr.Contains($"d{loop}", StringComparison.CurrentCulture);
                if (accessed)
                {
                    bufferMask.SetRelated(loop);
                    bufferTrips[pos] = bufferTrips[loop] * tileVars[loop];
                }
                else
                {
                    bufferTrips[pos] = bufferTrips[loop];
                }
            }

            // note update writes in second visitor.
        }

        var bufferInfo = new TileNodeBufferInfo<IntExpr>(bufferLiveness, accessMap, bufferPlaces, bufferShapes, bufferSizes, bufferTrips, bufferMask);
        return bufferInfo;
    }

    /// <summary>
    /// each buffer with each access Maps, note the access map domain is this node's domain. extents also mapping to current node's domain.
    /// </summary>
    /// <param name="BufferResults">buffer info.</param>
    /// <param name="DefUseMap">the defuse map is used to record cache buffer in the top memory level. </param>
    /// <param name="DimsMaps">dims map.</param>
    /// <param name="BackWardExtents"> backward extents for cout the buffer size. </param>
    public sealed record InitResult(BufferResult[] BufferResults, BiDictionary<BufferIdentity, BufferIdentity> DefUseMap, Dictionary<int, int>[] DimsMaps, IntExpr[][] BackWardExtents)
    {
    }

    /// <summary>
    /// buffer init result.
    /// </summary>
    /// <param name="Bid">buffer id.</param>
    /// <param name="Lifeness">buffer's lifetime.</param>
    /// <param name="AccessMap">access buffer relation from current node's domain, e.g. node.DomainRelation * buffer.AccessMap.</param>
    /// <param name="ElemSize">buffer size.</param>
    public sealed record BufferResult(BufferIdentity Bid, Tuple<int, int> Lifeness, AffineMap AccessMap, IntExpr ElemSize)
    {
    }

    public sealed record Context(int ParentOpId, IReadOnlyList<IntExpr> ForwardExtents, IntExpr TripCounts)
    {
        public static Context Default => new(-1, Array.Empty<IntVar>(), null!);
    }
}
