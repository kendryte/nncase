// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Google.OrTools.ConstraintSolver;
using NetFabric.Hyperlinq;
using Nncase.IR.Affine;

namespace Nncase.Schedule.TileTree;

public sealed class TreeSolverInitializer : TreeSolverBase, ITreeNodeVisitor<TreeSolverInitializer.Context, TreeSolverInitializer.InitResult>
{
    public TreeSolverInitializer(Solver solver, IntExpr one, IntExpr zero, IntExpr elem, Dictionary<OpNode, OpNodeInfo> primitiveBufferInfo, Dictionary<TileNode, TileNodeInfo> levelBufferInfos, Dictionary<ITileAbleNode, DomainInfo> domainDimInfos, ITargetOptions targetOptions)
        : base(solver, one, zero, elem, primitiveBufferInfo, levelBufferInfos, domainDimInfos, targetOptions)
    {
    }

    public InitResult Visit(ScopeNode value, Context context)
    {
        var bids = new List<BufferIdenitity>();
        var maps = new List<AffineMap>();
        var names = new List<Dictionary<int, int>>();
        var extents = new List<IntExpr[]>();
        for (int i = 0; i < value.Children.Count; i++)
        {
            var res = value.Children[i].Accept(this, context);
            bids.AddRange(res.Bids);
            maps.AddRange(res.AccessMaps);
            extents.AddRange(res.BackWardExtents);
            names.AddRange(res.DimsMaps);
        }

        return new(bids.ToArray(), maps.ToArray(), names.ToArray(), extents.ToArray());
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

        var defUseMap = GetBufferDefUseMap(childResult);

        // each tile node have buffer place vars.
        if (!TileNodeMemo.TryGetValue(value, out var info))
        {
            var bufferInfoMap = new Dictionary<BufferIdenitity, TileNodeBufferInfo>();
            for (int i = 0; i < childResult.Bids.Length; i++)
            {
                var sourceId = childResult.Bids[i];
                if (bufferInfoMap.ContainsKey(sourceId))
                {
                    continue;
                }

                if (defUseMap.TryGetValue(sourceId, out var sinkId))
                {
                    bufferInfoMap[sourceId] = bufferInfoMap[sinkId] = GetBufferInfo(value, sinkId, childResult.AccessMaps[childResult.Bids.IndexOf(sinkId)], backWardExtents);
                }
                else
                {
                    bufferInfoMap[sourceId] = GetBufferInfo(value, sourceId, childResult.AccessMaps[i], backWardExtents);
                }
            }

            TileNodeMemo.Add(value, new(backWardExtents, defUseMap, bufferInfoMap));
        }

        return new(childResult.Bids, childResult.AccessMaps.Select(m => value.DomainRelation.Map * m).ToArray(), new[] { dimsMap }, new[] { backWardExtents[0] });
    }

    public InitResult Visit(OpNode value, Context context)
    {
        var (pid, pvars) = context;
        var dimsMap = TreePrinter.GetDimsMap(value);
        var tileVars = value.DimNames.Select(n => Solver.MakeIntVar(1, long.MaxValue, $"{n}_L{value.Level}")).ToArray();
        var kernelInfo = CompilerServices.GetOpMicroKernelInfo(value.Op, value.AccessMaps[0].Domains.AsValueEnumerable().Select(i => i.Offset).ToArray(), value.AccessMaps.ToArray(), value.BufferShapes, TargetOptions);
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
        var resBids = new BufferIdenitity[value.ReadAccesses.Length + 1];
        var resRels = new AffineMap[value.ReadAccesses.Length + 1];

        for (int i = 0; i < value.ReadAccesses.Length; i++)
        {
            resBids[i] = new(value, i);
            resRels[i] = value.DomainRelation.Map * accessMaps[i];
        }

        resBids[value.ReadAccesses.Length] = new(value, value.ReadAccesses.Length);
        resRels[value.ReadAccesses.Length] = value.DomainRelation.Map * accessMaps[^1];
        return new(resBids, resRels, new[] { dimsMap }, new IntExpr[][] { tileVars.Cast<IntExpr>().ToArray() });
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

    /// <summary>
    /// source id => sink id.
    /// </summary>
    private Dictionary<BufferIdenitity, BufferIdenitity> GetBufferDefUseMap(InitResult childResult)
    {
        var map = new Dictionary<BufferIdenitity, BufferIdenitity>();
        for (int i = 0; i < childResult.Bids.Length; i++)
        {
            var sinkId = childResult.Bids[i];
            foreach (var dep in sinkId.Node.Dependences)
            {
                var sourceId = new BufferIdenitity(dep.Node, dep.Node.ReadAccesses.Length);
                if (childResult.Bids.IndexOf(sourceId) != -1)
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

    private TileNodeBufferInfo GetBufferInfo(TileNode tile, BufferIdenitity bid, AffineMap accessMap, IntExpr[][] backWardExtents)
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
            var subLevelPlace = bufferPlaces[i] = new IntVar[tile.Level + 1];
            for (int sl = 0; sl < tile.Level + 1; sl++)
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

        var bufferInfo = new TileNodeBufferInfo(accessMap, bufferPlaces, bufferShapes, bufferWrites, bufferSizeVars, bufferSizes, bufferMasks);
        return bufferInfo;
    }

    /// <summary>
    /// each buffer with each access Maps, note the access map domain is this node's domain. extents also mapping to current node's domain.
    /// </summary>
    public sealed record InitResult(BufferIdenitity[] Bids, AffineMap[] AccessMaps, Dictionary<int, int>[] DimsMaps, IntExpr[][] BackWardExtents)
    {
    }

    public sealed record Context(int ParentOpId, IReadOnlyList<IntExpr> ForwardExtents)
    {
        public static Context Default => new(-1, Array.Empty<IntVar>());
    }
}
