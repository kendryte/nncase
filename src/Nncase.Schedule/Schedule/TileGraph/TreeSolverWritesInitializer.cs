// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Reactive;
using Google.OrTools.ConstraintSolver;

namespace Nncase.Schedule.TileGraph;

public sealed class GraphSolverWritesInitializer : GraphSolverBase, ITileGraphVisitor<Dictionary<BufferIdentity, IntExpr[]>, Unit>
{
    public GraphSolverWritesInitializer(Solver solver, Dictionary<OpNode, OpNodeInfo> primitiveBufferInfo, Dictionary<TileGraph, TileNodeInfo> levelBufferInfos, Dictionary<ITileableNode, DomainInfo> domainDimInfos, ITargetOptions targetOptions)
        : base(solver, primitiveBufferInfo, levelBufferInfos, domainDimInfos, targetOptions)
    {
    }

    /// <summary>
    /// buffer trip counts mean each buffer's trip count at loop i.
    /// </summary>
    public Unit Visit(TileGraph value, Dictionary<BufferIdentity, IntExpr[]> bufferTripCounts)
    {
        Dictionary<BufferIdentity, IntExpr[]> currentTripCounts = new();
        var domainInfo = TileableNodeMemo[value];
        if (value.Parent is TileGraph parentTileNode)
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

        if (value.ClustersCount == 0)
        {
            foreach (var item in value.Clusters.OfType<TileGraph>())
            {
                Visit(item, currentTripCounts);
            }
        }
        else
        {
            foreach (var item in value.Vertices)
            {
                Visit(item, currentTripCounts);
            }
        }

        return default;
    }

    public Unit Visit(OpNode value, Dictionary<BufferIdentity, IntExpr[]> bufferTripCounts)
    {
        return default;
    }
}
