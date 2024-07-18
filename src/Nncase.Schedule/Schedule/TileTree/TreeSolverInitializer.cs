// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Google.OrTools.ConstraintSolver;
using NetFabric.Hyperlinq;
using Nncase.IR.Affine;

namespace Nncase.Schedule.TileTree;

public sealed class TreeSolverInitializer : TreeSolverBase, ITreeNodeVisitor<TreeSolverInitializer.Context, TreeSolverInitializer.InitResult>
{
    public TreeSolverInitializer(int totalLevel, Solver solver, IntExpr one, IntExpr zero, IntExpr elem, Dictionary<OpNode, OpNodeInfo> primitiveBufferInfo, Dictionary<TileNode, TileNodeInfo> levelBufferInfos, Dictionary<ITileAbleNode, DomainInfo> domainDimInfos, ITargetOptions targetOptions)
        : base(solver, one, zero, elem, primitiveBufferInfo, levelBufferInfos, domainDimInfos, targetOptions)
    {
        TotalLevel = totalLevel;
    }

    public int TimeStamp { get; private set; }

    public int TotalLevel { get; }

    /// <summary>
    /// source id => sink id.
    /// </summary>
    public static Dictionary<BufferIdenitity, BufferIdenitity> GetBufferDefUseMap(BufferResult[] bufferResults)
    {
        var map = new Dictionary<BufferIdenitity, BufferIdenitity>();
        for (int i = 0; i < bufferResults.Length; i++)
        {
            var sinkId = bufferResults[i].Bid;
            foreach (var dep in sinkId.Node.Dependences)
            {
                var sourceId = new BufferIdenitity(dep.Node, dep.Node.ReadAccesses.Length);
                if (Array.FindIndex(bufferResults, r => r.Bid == sourceId) != -1)
                {
                    if (!map.ContainsKey(sourceId))
                    {
                        map.Add(sourceId, sinkId);
                    }
                }
            }
        }

        return map;
    }

    public static ArgumentsInfo GetArgumentsInfo(BufferResult[] bufferResults)
    {
        var map = GetBufferDefUseMap(bufferResults);
        var inputs = new HashSet<BufferIdenitity>(bufferResults.Select(b => b.Bid).Where(b => b.Index != b.Node.BufferShapes.Length - 1));
        var outputs = new HashSet<BufferIdenitity>(bufferResults.Select(b => b.Bid).Where(b => b.Index == b.Node.BufferShapes.Length - 1));

        foreach (var (k, v) in map)
        {
            inputs.Remove(k);
            inputs.Remove(v);
            outputs.Remove(k);
            outputs.Remove(v);
        }

        return new(inputs, outputs, map);
    }

    public InitResult Visit(ScopeNode value, Context context)
    {
        var results = new List<BufferResult>();
        var names = new List<Dictionary<int, int>>();
        var extents = new List<IntExpr[]>();
        for (int i = 0; i < value.Children.Count; i++)
        {
            var res = value.Children[i].Accept(this, context);
            results.AddRange(res.BufferResults);
            extents.AddRange(res.BackWardExtents);
            names.AddRange(res.DimsMaps);
        }

        return new(results.ToArray(), names.ToArray(), extents.ToArray());
    }

    public InitResult Visit(TileNode value, Context context)
    {
        var (pid, pvars) = context;
        var dimsMap = TreePrinter.GetDimsMap(value);
        if (!pvars.Any())
        {
            dimsMap.Clear();
        }

        var tileVars = value.DimNames.Select(n => Solver.MakeIntVar(1, int.MaxValue, $"{n}_L{value.Level}")).ToArray();
        var forwardExtents = tileVars.Cast<IntExpr>().ToArray();
        if (!TileableNodeMemo.TryGetValue(value, out var dimInfo))
        {
            foreach (var (k, v) in dimsMap)
            {
                forwardExtents[k] *= pvars[v];
            }

            TileableNodeMemo.Add(value, new(tileVars, forwardExtents, dimsMap));
        }

        var childResult = value.Child.Accept(this, context with { ParentOpId = value.OpId, ForwardExtents = forwardExtents });

        var backWardExtents = GetBackWardExtents(tileVars, childResult.DimsMaps, childResult.BackWardExtents);

        var defUseMap = GetBufferDefUseMap(childResult.BufferResults);
        var bufferResults = new List<BufferResult>();

        // each tile node have buffer place vars.
        if (!TileNodeMemo.TryGetValue(value, out var info))
        {
            var bufferInfoMap = new Dictionary<BufferIdenitity, TileNodeBufferInfo>();
            for (int i = 0; i < childResult.BufferResults.Length; i++)
            {
                var result = childResult.BufferResults[i];
                BufferIdenitity currentId;
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

        return new(bufferResults.ToArray(), new[] { dimsMap }, new[] { backWardExtents[0] });
    }

    public InitResult Visit(OpNode value, Context context)
    {
        var (pid, pvars) = context;
        var dimsMap = TreePrinter.GetDimsMap(value);
        var tileVars = value.DimNames.Select(n => Solver.MakeIntVar(1, long.MaxValue, $"{n}_L{value.Level}")).ToArray();

        // CompilerServices.GetOpMicroKernelInfo(value.Op, value.AccessMaps[0].Domains.AsValueEnumerable().Select(i => i.Offset).ToArray(), value.AccessMaps.ToArray(), value.BufferShapes, TargetOptions);
        var kernelInfo = new MicroKernelInfo(tileVars.Select(i => 1).ToArray(), tileVars.Select((_, i) => new ValueRange<int>(0, value.DomainBounds[i])).ToArray(), 1, 1);

        for (int i = 0; i < tileVars.Length; i++)
        {
            tileVars[i].SetRange(kernelInfo.Multiplier[i].Min, kernelInfo.Multiplier[i].Max);
        }

        var primtiveMap = AffineMap.FromCallable((doms, syms) => doms.Select(i => new AffineRange(i.Offset, kernelInfo.Primitives[i.Extent.Position] * i.Extent)).ToArray(), value.DomainBounds.Count);
        var accessMaps = new AffineMap[value.BufferShapes.Length];

        // cache the primitive buffer shape and sizes.
        if (!OpNodeMemo.TryGetValue(value, out var info))
        {
            var shapes = new IntExpr[value.BufferShapes.Length][];
            var sizes = new IntExpr[value.BufferShapes.Length];
            for (int a = 0; a < value.BufferShapes.Length; a++)
            {
                shapes[a] = new IntExpr[value.BufferShapes[a].Length];
                sizes[a] = Elem;
                var extentVars = tileVars;
                var converter = new AffineExprToIntExprConverter(Solver, extentVars);
                accessMaps[a] = primtiveMap * value.AccessMaps[a];
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

        for (int i = 0; i < value.ReadAccesses.Length; i++)
        {
            bufferResults[i] = new(new(value, i), new(TimeStamp, TimeStamp + 1), value.DomainRelation.Map * accessMaps[i]);
        }

        bufferResults[value.ReadAccesses.Length] = new(new(value, value.ReadAccesses.Length), new(TimeStamp, TimeStamp + 1), value.DomainRelation.Map * accessMaps[^1]);
        TimeStamp += 2;

        // todo backward extents should times primtives.
        return new(bufferResults, new[] { dimsMap }, new IntExpr[][] { tileVars.Cast<IntExpr>().ToArray() });
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

    private TileNodeBufferInfo GetBufferInfo(TileNode tile, BufferIdenitity bid, AffineMap accessMap, Tuple<int, int> lifeness, IntExpr[][] backWardExtents)
    {
        var domainDims = tile.DimNames.Length;
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

            bufferSizes[i] = subDomainShapes.Aggregate(Elem, Solver.MakeProd);
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
    public sealed record InitResult(BufferResult[] BufferResults, Dictionary<int, int>[] DimsMaps, IntExpr[][] BackWardExtents)
    {
    }

    public sealed record BufferResult(BufferIdenitity Bid, Tuple<int, int> Lifeness, AffineMap AccessMap)
    {
    }

    public sealed record Context(int ParentOpId, IReadOnlyList<IntExpr> ForwardExtents)
    {
        public static Context Default => new(-1, Array.Empty<IntVar>());
    }
}
