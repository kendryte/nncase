// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections;
using System.Diagnostics.CodeAnalysis;
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
/// the placement length = domain rank + 1. if domain dims = 4, create loop will be 0,1,2,3,4.
/// create loop = 0 means we create buffer in outside of all loops.
/// for example, create loop = 2, means create buffer d0,d1,(buffer create here) d2,d3.
/// </summary>
/// <param name="Liveness">this buffer's liveness.</param>
/// <param name="Map">this buffers access map.</param>
/// <param name="Places">
/// Places[create loop][store level]:
/// create loop in [0, domain rank] , 0 means out all, 1 means out loop0, domain rank means in loopN.
/// store level in [0, create level == top level ? create level : top level - 1), 0 means level 1, 1 means level 2. </param>
/// <param name="Shapes">the buffer shape according to the placement.</param>
/// <param name="SizeVars">the buffer size according to the placement.</param>
/// <param name="SizeExprs">the buffer size expr.</param>
/// <param name="Trips">related loop trips.</param>
/// <param name="Masks">record this tile's loop vars which involved buffer size.</param>
public sealed record TileNodeBufferInfo<T>(Tuple<int, int> Liveness, AffineMap Map, T[][] Places, T[][] Shapes, T[] SizeVars, T[] SizeExprs, T[] Trips, LoopMask[] Masks)
{
}

/// <summary>
/// the placement length = domain rank + 1. if domain dims = 4, create loop will be 0,1,2,3,4.
/// create loop = 0 means we create buffer in outside of all loops.
/// for example, create loop = 2, means create buffer d0,d1,(buffer create here) d2,d3.
/// </summary>
/// <param name="TripCounts">forward trips, length = domainRank+1. the trips[i] means trips accumulated until loop var[i].</param>
/// <param name="BackWardExtents">accumulated backward extents.</param>
/// <param name="DefUseMap">key is def, value is use.</param>
/// <param name="BufferInfoMap">buffer info memo.</param>
public sealed record TileNodeInfo<T>(T[] TripCounts, T[][] BackWardExtents, Dictionary<BufferIdentity, BufferIdentity> DefUseMap, Dictionary<BufferIdentity, TileNodeBufferInfo<T>> BufferInfoMap)
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

    public bool TryGetBufferInfo(BufferIdentity bid, [MaybeNullWhen(false)] out TileNodeBufferInfo<T> info)
    {
        if (DefUseMap.TryGetValue(bid, out var sinkId))
        {
            BufferInfoMap.TryGetValue(sinkId, out info);
        }
        else
        {
            BufferInfoMap.TryGetValue(bid, out info);
        }

        return info is not null;
    }
}

/// <summary>
/// domain infomation.
/// </summary>
/// <param name="TileVars">loop trip vars length = domainRank.</param>
/// <param name="ForwardExtents">forward extents.</param>
/// <param name="DimsMap"> key is current dim, value is partent dim. </param>
public sealed record DomainInfo<T>(T[] TileVars, T[] ForwardExtents, Dictionary<int, int> DimsMap)
{
}

/// <summary>
/// op node info.
/// </summary>
/// <param name="Maps">current node's domain accesses the buffer. it means applyed by this op's domain relation.</param>
/// <param name="Shapes">each buffer's shape expr. </param>
/// <param name="Sizes">each buffer's size.</param>
public sealed record OpNodeInfo<T>(AffineMap[] Maps, T[][] Shapes, T[] Sizes)
{
}
