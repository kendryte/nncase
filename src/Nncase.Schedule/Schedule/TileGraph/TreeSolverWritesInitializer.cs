// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Reactive;
using Google.OrTools.ConstraintSolver;

namespace Nncase.Schedule.TileGraph;

public sealed class TreeSolverWritesInitializer : TreeSolverBase<IntExpr>, ITreeNodeVisitor<Dictionary<BufferIdentity, IntExpr[]>, Unit>
{
    public TreeSolverWritesInitializer(Solver solver, Dictionary<OpNode, OpNodeInfo<IntExpr>> primitiveBufferInfo, Dictionary<TileNode, TileNodeInfo<IntExpr>> levelBufferInfos, Dictionary<ITileable, DomainInfo<IntExpr>> domainDimInfos, ITargetOptions targetOptions)
        : base(solver, primitiveBufferInfo, levelBufferInfos, domainDimInfos, targetOptions)
    {
    }

    /// <summary>
    /// buffer trip counts mean each buffer's trip count at loop i.
    /// </summary>
    public Unit Visit(TileNode value, Dictionary<BufferIdentity, IntExpr[]> bufferTripCounts)
    {
        Dictionary<BufferIdentity, IntExpr[]> currentTripCounts = new();
        var domainInfo = TileableNodeMemo[value];

        // for child graph node.
        if (value.Parent is TileNode parentTileNode && parentTileNode.OpId != -1)
        {
            var partentTileInfo = TileNodeMemo[parentTileNode];

            // 1. child domain map to parent domain.
            foreach (var (bid, bufferInfo) in TileNodeMemo[value].BufferInfoMap)
            {
                var parentTripCounts = bufferTripCounts[partentTileInfo.GetCacheBid(bid)];
                var tripCounts = new IntExpr[domainInfo.TileVars.Length];

                for (int i = 0; i < domainInfo.TileVars.Length; i++)
                {
                    IntExpr factor;
                    IntExpr parentFactor;
                    if (bufferInfo.Masks[i].IsRelated(i))
                    {
                        factor = domainInfo.TileVars[i];
                    }
                    else
                    {
                        factor = Solver.MakeIntConst(1);
                    }

                    if (domainInfo.DimsMap.TryGetValue(i, out var j))
                    {
                        parentFactor = parentTripCounts[j];
                    }
                    else
                    {
                        parentFactor = Solver.MakeIntConst(1);
                    }

                    tripCounts[i] = factor * parentFactor;
                    bufferInfo.Writes[i] = bufferInfo.SizeVars[i] * tripCounts[i];
                }

                currentTripCounts.Add(bid, tripCounts);
            }
        }
        else
        {
            // for prim graph node.
            foreach (var (bid, bufferInfo) in TileNodeMemo[value].BufferInfoMap)
            {
                var tripCounts = new IntExpr[domainInfo.TileVars.Length];

                for (int i = 0; i < domainInfo.TileVars.Length; i++)
                {
                    IntExpr factor;
                    if (bufferInfo.Masks[i].IsRelated(i))
                    {
                        factor = domainInfo.TileVars[i];
                    }
                    else
                    {
                        factor = Solver.MakeIntConst(1);
                    }

                    tripCounts[i] = factor;
                    bufferInfo.Writes[i] = bufferInfo.SizeVars[i] * tripCounts[i];
                }

                currentTripCounts.Add(bid, tripCounts);
            }
        }

        foreach (var item in value.Children)
        {
            item.Accept(this, currentTripCounts);
        }

        return default;
    }

    public Unit Visit(OpNode value, Dictionary<BufferIdentity, IntExpr[]> bufferTripCounts)
    {
        return default;
    }
}
