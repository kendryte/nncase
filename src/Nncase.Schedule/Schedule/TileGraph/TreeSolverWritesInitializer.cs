// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Reactive;
using Google.OrTools.ConstraintSolver;

namespace Nncase.Schedule.TileGraph;

public sealed class TreeSolverWritesInitializer : TreeSolverBase<IntExpr>, ITreeNodeVisitor<Dictionary<BufferIdentity, IntExpr>, Unit>
{
    public TreeSolverWritesInitializer(int topLevel, Solver solver, Dictionary<OpNode, OpNodeInfo<IntExpr>> primitiveBufferInfo, Dictionary<TileNode, TileNodeInfo<IntExpr>> levelBufferInfos, Dictionary<ITileable, DomainInfo<IntExpr>> domainDimInfos, ICpuTargetOptions targetOptions)
        : base(solver, primitiveBufferInfo, levelBufferInfos, domainDimInfos, targetOptions)
    {
        TopLevel = topLevel;
    }

    public int TopLevel { get; }

    public static void Init(ITreeNode tree, int topLevel, Solver solver, Dictionary<OpNode, OpNodeInfo<IntExpr>> primitiveBufferInfo, Dictionary<TileNode, TileNodeInfo<IntExpr>> levelBufferInfos, Dictionary<ITileable, DomainInfo<IntExpr>> domainDimInfos, ICpuTargetOptions targetOptions)
    {
        var initzer = new TreeSolverWritesInitializer(topLevel, solver, primitiveBufferInfo, levelBufferInfos, domainDimInfos, targetOptions);
        tree.Accept(initzer, new());
    }

    /// <summary>
    /// buffer trip counts mean each buffer's trip count at loop i.
    /// </summary>
    public Unit Visit(TileNode value, Dictionary<BufferIdentity, IntExpr> bufferTripCounts)
    {
        Dictionary<BufferIdentity, IntExpr> currentTripCounts = new();
        var domainInfo = TileableNodeMemo[value];
        TileNodeInfo<IntExpr>? partentTileInfo = null;
        if (value.Parent is TileNode { Level: not -1 } parentNode)
        {
            partentTileInfo = TileNodeMemo[parentNode];
        }

        // 1. child domain map to parent domain.
        foreach (var (bid, bufferInfo) in TileNodeMemo[value].BufferInfoMap)
        {
            var parentTripCounts = partentTileInfo is null ? Solver.MakeIntConst(1) : bufferTripCounts[partentTileInfo.GetCacheBid(bid)];

            for (int i = 0; i < domainInfo.TileVars.Length + 1; i++)
            {
                IntExpr trip = parentTripCounts;
                for (int j = 0; j < domainInfo.TileVars.Length; j++)
                {
                    if (bufferInfo.Masks[i].IsRelated(j))
                    {
                        trip = trip * domainInfo.TileVars[j];
                    }
                }

                bufferInfo.Trips[i] = trip;
            }

            currentTripCounts.Add(bid, bufferInfo.Trips[^1]);
        }

        foreach (var item in value.Children)
        {
            item.Accept(this, currentTripCounts);
        }

        return default;
    }

    public Unit Visit(OpNode value, Dictionary<BufferIdentity, IntExpr> bufferTripCounts)
    {
        return default;
    }
}
