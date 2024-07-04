// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections;
using Google.OrTools.ConstraintSolver;
using Nncase.IR;
using VisitorPatternGenerator;

namespace Nncase.Schedule.TileTree;

public sealed record BufferIdenitity(OpNode Op, int Index)
{
    public override string ToString() => $"Op{Op.OpId}_{Index}";
}

public sealed record TileNodeBufferAssignment(bool[][] Place, long[] Write, long[] Size)
{
}

public sealed record TileNodeAssignment(Dictionary<BufferIdenitity, BufferIdenitity> DefUseMap, Dictionary<BufferIdenitity, TileNodeBufferAssignment> BufferInfoMap)
{
}

public sealed record DomainDimAssignment(long[] TileVars)
{
}

/// <summary>
/// Place : [create_loop,store_level], Shapes: [create_loop][shape] write: [create_loop], size: [create loop].
/// </summary>
public sealed record TileNodeBufferInfo(IntVar[][] Places, IntExpr[][] Shapes, IntExpr[] Writes, IntExpr[] Sizes, LoopMask[] Masks)
{
}

public sealed record TileNodeInfo(IntExpr[][] DomainExtents, Dictionary<BufferIdenitity, BufferIdenitity> DefUseMap, Dictionary<BufferIdenitity, TileNodeBufferInfo> BufferInfoMap)
{
}

/// <summary>
/// loop masks.count == buffer.dimension.
/// </summary>
public sealed record DomainInfo(IntVar[] TileVars, IntExpr[] ForwardExtents, Dictionary<int, int> DimsMap)
{
}

public sealed record OpNodeInfo(IntExpr[][] Shapes, IntExpr[] Size)
{
}
