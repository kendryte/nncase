// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections;
using Google.OrTools.ConstraintSolver;
using Nncase.IR;
using Nncase.IR.Affine;

namespace Nncase.Schedule.TileGraph;

public sealed record BufferIdentity(TileGrid Node, int Index)
{
    public bool IsOutput => Index == Node.ReadAccesses.Length;

    public override string ToString() => IsOutput ? $"Op{Node.OpId}_Out" : $"Op{Node.OpId}_in{Index}";
}

/// <summary>
/// Map: current offset/extent  Place : [create_loop,store_level], Shapes: [create_loop][shape] write: [create_loop], size: [create loop].
/// </summary>
public sealed record TileNodeBufferInfo<T>(Tuple<int, int> Liveness, AffineMap Map, T[][] Places, T[][] Shapes, T[] Writes, T[] SizeVars, T[] SizeExprs, LoopMask[] Masks)
{
}

public sealed record TileNodeInfo<T>(T[][] BackWardExtents, Dictionary<BufferIdentity, BufferIdentity> DefUseMap, Dictionary<BufferIdentity, TileNodeBufferInfo<T>> BufferInfoMap)
{
    public BufferIdentity GetCacheBid(BufferIdentity bid)
    {
        if (DefUseMap.TryGetValue(bid, out var sinkId))
        {
            return sinkId;
        }

        if (!BufferInfoMap.ContainsKey(bid))
        {
            throw new KeyNotFoundException(bid.ToString());
        }

        return bid;
    }
}

/// <summary>
/// loop masks.count == buffer.dimension. the dims map is current dims map to partent dims.
/// </summary>
public sealed record DomainInfo<T>(T[] TileVars, T[] ForwardExtents, Dictionary<int, int> DimsMap)
{
}

/// <summary>
/// op node info.
/// </summary>
/// <param name="Maps">current node's domain accesses the buffer. it means applyed by this op's domain relation.</param>
/// <param name="Shapes">this buffer's shape expr. </param>
/// <param name="Size">buffer's size.</param>
public sealed record OpNodeInfo<T>(AffineMap[] Maps, T[][] Shapes, T[] Size)
{
}
