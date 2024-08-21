// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections;
using Google.OrTools.ConstraintSolver;
using Nncase.IR;
using Nncase.IR.Affine;

namespace Nncase.Schedule.TileGraph;

public sealed record BufferIdentity(OpNode Node, int Index)
{
    public bool IsOutput => Index == Node.ReadAccesses.Length;

    public override string ToString() => $"Op{Node.OpId}_{Index}";
}

/// <summary>
/// Map: current offset/extent  Place : [create_loop,store_level], Shapes: [create_loop][shape] write: [create_loop], size: [create loop].
/// </summary>
public sealed record TileNodeBufferInfo(Tuple<int, int> Liveness, AffineMap Map, IntVar[][] Places, IntExpr[][] Shapes, IntExpr[] Writes, IntExpr[] SizeVars, IntExpr[] SizeExprs, LoopMask[] Masks)
{
}

public sealed record TileNodeInfo(IntExpr[][] BackWardExtents, Dictionary<BufferIdentity, BufferIdentity> DefUseMap, Dictionary<BufferIdentity, TileNodeBufferInfo> BufferInfoMap)
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
public sealed record DomainInfo(IntVar[] TileVars, IntExpr[] ForwardExtents, Dictionary<int, int> DimsMap)
{
}

/// <summary>
/// op node info.
/// </summary>
/// <param name="Maps">current node's domain accesses the buffer. it means applyed by this op's domain relation.</param>
/// <param name="Shapes">this buffer's shape expr. </param>
/// <param name="Size">buffer's size.</param>
public sealed record OpNodeInfo(AffineMap[] Maps, IntExpr[][] Shapes, IntExpr[] Size)
{
}

public sealed record ArgumentsInfo(HashSet<BufferIdentity> Inputs, HashSet<BufferIdentity> Outputs, Dictionary<BufferIdentity, BufferIdentity> DefUseMap)
{
    public enum BufferKind
    {
        Input,
        Output,

        /// <summary>
        /// cache in the kernel's outside.
        /// </summary>
        Cache,

        /// <summary>
        /// cache in the kernel's inside.
        /// </summary>
        None,
    }

    public BufferIdentity GetUniqueIdenitity(BufferIdentity bid)
    {
        if (Inputs.Contains(bid))
        {
            return bid;
        }
        else if (Outputs.Contains(bid))
        {
            return bid;
        }
        else if (DefUseMap.ContainsKey(bid))
        {
            return DefUseMap[bid];
        }
        else if (DefUseMap.ContainsValue(bid))
        {
            return bid;
        }

        throw new NotSupportedException();
    }

    public BufferKind GetBufferKind(BufferIdentity bid)
    {
        if (Inputs.Contains(bid))
        {
            return BufferKind.Input;
        }
        else if (Outputs.Contains(bid))
        {
            return BufferKind.Output;
        }
        else if (DefUseMap.ContainsKey(bid) || DefUseMap.ContainsValue(bid))
        {
            return BufferKind.Cache;
        }

        return BufferKind.None;
    }
}
