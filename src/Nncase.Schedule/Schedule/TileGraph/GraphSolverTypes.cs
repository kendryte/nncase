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

    public bool IsOutputLiveOut => IsOutput && Node.Attribute.HasFlag(TileGridAttribute.LiveOut);

    public override string ToString() => IsOutput ? $"Op{Node.OpId}_Out" : $"Op{Node.OpId}_in{Index}";
}

/// <summary>
/// the placement length = domain rank + 1. if domain dims = 4, create loop will be 0,1,2,3,4.
/// create loop = 0 means we create buffer in outside of all loops.
/// for example, create loop = 2, means create buffer d0,d1,(buffer create here) d2,d3.
/// </summary>
/// <param name="Liveness">this buffer's liveness for each create loop.</param>
/// <param name="Map">this buffers access map.</param>
/// <param name="Places">
/// Places[create loop][store level]:
/// create loop in [0, domain rank] , 0 means out all, 1 means out loop0, domain rank means in loopN.
/// note only the nodes which store at top level have valid Places[0], else the Places[0] is empty.
/// store level in [0, create level == top level ? create level : top level - 1), 0 means level 1, 1 means level 2. </param>
/// <param name="Shapes">the buffer shape according to the placement.</param>
/// <param name="Sizes">the buffer size according to the placement.</param>
/// <param name="Trips">related loop trips at current domain.</param>
/// <param name="Mask">the loop mask of this buffer at current domain.</param>
public sealed record TileNodeBufferInfo<T>(Tuple<int, int>[] Liveness, AffineMap Map, T[][] Places, T[][] Shapes, T[] Sizes, T[] Trips, LoopMask Mask)
{
    public int GetLastRelatedPos()
    {
        var lastLoop = Mask.LastRelated(Places.Length - 1);
        return lastLoop + 1;
    }
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
public sealed record TileNodeInfo<T>(T[] TripCounts, T[][] BackWardExtents, BiDictionary<BufferIdentity, BufferIdentity> DefUseMap, Dictionary<BufferIdentity, TileNodeBufferInfo<T>> BufferInfoMap)
{
    public BufferIdentity GetByChildBuffer(BufferIdentity sinkBid)
    {
        if (DefUseMap.TryGetByValue(sinkBid, out var sourceBid))
        {
            return sourceBid;
        }

        if (!BufferInfoMap.ContainsKey(sinkBid))
        {
            throw new KeyNotFoundException(sinkBid.ToString());
        }

        return sinkBid;
    }
}

/// <summary>
/// domain infomation.
/// </summary>
/// <param name="TileVars">loop trip vars length = domainRank.</param>
/// <param name="ForwardExtents">forward extents.</param>
/// <param name="DimsMap"> key is current dim, value is partent dim. </param>
/// NOTE dimsMap should be removed when using isl.
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

public class BiDictionary<TKey, TValue> : IEnumerable<KeyValuePair<TKey, TValue>>, IEnumerable
    where TKey : notnull
    where TValue : notnull
{
    private readonly Dictionary<TKey, HashSet<TValue>> _forward = new();

    private readonly Dictionary<TValue, TKey> _reverse = new();

    public bool TryGetByKey(TKey key, [MaybeNullWhen(false)] out HashSet<TValue> value) => _forward.TryGetValue(key, out value);

    public bool TryGetByValue(TValue value, [MaybeNullWhen(false)] out TKey key) => _reverse.TryGetValue(value, out key);

    public bool Add(TKey key, TValue value)
    {
        if (!_forward.TryGetValue(key, out var values))
        {
            values = new() { };
            _forward[key] = values;
        }

        if (values.Contains(value))
        {
            return false;
        }

        values.Add(value);
        _reverse[value] = key;
        return true;
    }

    public bool ContainsKey(TKey value) => _forward.ContainsKey(value);

    public bool ContainsValue(TValue value) => _reverse.ContainsKey(value);

    public IEnumerator<KeyValuePair<TKey, TValue>> GetEnumerator()
    {
        foreach (var kv in _forward)
        {
            foreach (var value in kv.Value)
            {
                yield return new KeyValuePair<TKey, TValue>(kv.Key, value);
            }
        }
    }

    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
}
