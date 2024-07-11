// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections;
using Google.OrTools.ConstraintSolver;
using Nncase.IR;
using Nncase.IR.Affine;
using VisitorPatternGenerator;

namespace Nncase.Schedule.TileTree;

public sealed record BufferIdenitity(OpNode Node, int Index)
{
    public override string ToString() => $"Op{Node.OpId}_{Index}";
}

/// <summary>
/// Map: current offset/extent  Place : [create_loop,store_level], Shapes: [create_loop][shape] write: [create_loop], size: [create loop].
/// </summary>
public sealed record TileNodeBufferInfo(ValueRange<int> Lifeness, AffineMap Map, IntVar[][] Places, IntExpr[][] Shapes, IntExpr[] Writes, IntExpr[] SizeVars, IntExpr[] SizeExprs, LoopMask[] Masks)
{
}

public sealed record TileNodeInfo(IntExpr[][] BackWardExtents, Dictionary<BufferIdenitity, BufferIdenitity> DefUseMap, Dictionary<BufferIdenitity, TileNodeBufferInfo> BufferInfoMap)
{
    public BufferIdenitity GetCacheBid(BufferIdenitity bid)
    {
        if (DefUseMap.TryGetValue(bid, out var sinkId))
        {
            return sinkId;
        }

        return bid;
    }
}

/// <summary>
/// loop masks.count == buffer.dimension. the dims map is current dims map to partent dims.
/// </summary>
public sealed record DomainInfo(IntVar[] TileVars, IntExpr[] ForwardExtents, Dictionary<int, int> DimsMap)
{
}

public sealed record OpNodeInfo(AffineMap[] Maps, IntExpr[][] Shapes, IntExpr[] Size)
{
}
