// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using NetFabric.Hyperlinq;
using Nncase.IR;

namespace Nncase.Utilities;

public static class DistributedUtility
{
    public static IReadOnlyList<IRArray<SBP>> GetLeafCandidateNDSBPs(TensorType tensorType, Placement placement)
    {
        var ndsbps = new List<List<SBP>>();
        for (int i = 0; i < placement.Rank; i++)
        {
            var ndsbp = new List<SBP>();
            if (tensorType.Shape.All(x => x.IsDynamic || x.FixedValue != 0))
            {
                for (int axis = 0; axis < tensorType.Shape.Rank; axis++)
                {
                    if (tensorType.Shape[axis] is { IsFixed: true, FixedValue: long s } && placement.Hierarchy[i] > 1 && IsDivideExactly(s, placement.Hierarchy[i]))
                    {
                        ndsbp.Add(SBP.S(axis));
                    }
                }
            }

            ndsbp.Add(SBP.B);
            ndsbps.Add(ndsbp);
        }

        return ndsbps.CartesianProduct().Select(ndsbp => ndsbp.ToArray()).Where(ndsbp => IsDistributable(tensorType, ndsbp, placement)).Select(ndsbp => new IRArray<SBP>(ndsbp)).ToArray();
    }

    public static bool IsDistributable(TensorType tensorType, ReadOnlySpan<SBP> ndsbp, Placement placement)
    {
        if (!tensorType.Shape.IsRanked)
        {
            return false;
        }

        var divisors = GetDivisors(new DistributedType(tensorType, new IRArray<SBP>(ndsbp.ToArray()), placement));
        return divisors.Select((d, axis) => (d, axis)).All(p => p.d == 0 ? true : IsDivideExactly(tensorType.Shape[p.axis].FixedValue, p.d));
    }

    public static IReadOnlyList<int> GetDivisors(DistributedType distributedType)
    {
        var rank = distributedType.TensorType.Shape.Rank;
        var divisors = Enumerable.Repeat(0, rank).ToArray();
        for (int i = 0; i < distributedType.NdSBP.Count; i++)
        {
            if (distributedType.NdSBP[i] is SBPSplit { Axis: int axis })
            {
                if (divisors[axis] == 0)
                {
                    divisors[axis] = 1;
                }

                divisors[axis] *= distributedType.Placement.Hierarchy[i];
            }
        }

        return divisors;
    }

    public static bool TryGetDividedTensorType(DistributedType distributedType, [System.Diagnostics.CodeAnalysis.MaybeNullWhen(false)] out TensorType tensorType)
    {
        tensorType = null;
        var divisors = GetDivisors(distributedType);
        tensorType = new TensorType(
            distributedType.TensorType.DType,
            distributedType.TensorType.Shape.Zip(divisors).Select(p => p.Second == 0 ? p.First : Dimension.CeilDiv(p.First, p.Second)).ToArray());
        return true;
    }

    public static Expr[] TryGetNonUniformDividedShape(DistributedType distributedType)
    {
        var shape = distributedType.TensorType.Shape.ToValueArray();
        var hierarchies = Enumerable.Range(0, shape.Length).Select(i => new List<int>()).ToArray();
        var ids = distributedType.Placement.Name.Select(c => new Var(c + "id", TensorType.Scalar(DataTypes.Int32))).ToArray();
        var hierarchyStrides = TensorUtilities.GetStrides(distributedType.Placement.Hierarchy.ToArray());
        for (int i = 0; i < distributedType.NdSBP.Count; i++)
        {
            if (distributedType.NdSBP[i] is SBPSplit { Axis: int axis })
            {
                hierarchies[axis].Add(i);
            }
        }

        return hierarchies.Select((divs, axis) =>
        {
            Expr dim;
            if (divs.Any())
            {
                var divsor = (int)TensorUtilities.GetProduct(divs.Select(h => distributedType.Placement.Hierarchy[h]).ToArray());
                var (res, rem) = Math.DivRem(shape[axis], divsor);
                if (rem == 0)
                {
                    return res;
                }

                dim = IR.F.Math.Select(
                    TensorUtilities.GetIndex(hierarchyStrides.TakeLast(divs.Count).Select(s => (Expr)s).ToArray(), divs.Select(h => ids[h]).ToArray()) < (divsor - 1),
                    res,
                    res + rem);
            }
            else
            {
                dim = distributedType.TensorType.Shape[axis].FixedValue;
            }

            return dim;
        }).ToArray();
    }

    public static List<long[]> TryGetNonUniformDividedSlice(DistributedType distributedType)
    {
        var shape = distributedType.TensorType.Shape.ToValueArray();
        var hierarchies = Enumerable.Range(0, shape.Length).Select(i => new List<int>()).ToArray();
        for (int i = 0; i < distributedType.NdSBP.Count; i++)
        {
            if (distributedType.NdSBP[i] is SBPSplit { Axis: int axis })
            {
                hierarchies[axis].Add(i);
            }
        }

        var spliList = hierarchies.Select<List<int>, long[]>((divs, axis) =>
        {
            long[] dim;
            if (divs.Any())
            {
                var divsor = (int)TensorUtilities.GetProduct(divs.Select(h => distributedType.Placement.Hierarchy[h]).ToArray());
                var (res, rem) = Math.DivRem(shape[axis], divsor);
                if (rem == 0)
                {
                    return new[] { res };
                }

                dim = new[] { res, res + rem };
            }
            else
            {
                dim = distributedType.TensorType.Shape.ToValueArray().Skip(axis).Take(1).ToArray();
            }

            return dim;
        }).ToList();

        IEnumerable<long[]> ret = new[] { Array.Empty<long>() };
        foreach (long[] array in spliList)
        {
            ret = from seq in ret
                  from item in array
                  select seq.Concat(new[] { item }).ToArray();
        }

        return ret.ToList();
    }

    public static bool IsDivideBy(long input, int divisor)
    {
        if (input >= divisor)
        {
            return true;
        }

        return false;
    }

    public static bool IsDivideExactly(long input, int divisor)
    {
        if (input >= divisor && input % divisor == 0)
        {
            return true;
        }

        return false;
    }

    public static float GetDividedTensorEfficiency(DistributedType distributedType, int burstLength)
    {
        var (tiles, shape) = GetDividedTile(distributedType);
        if (tiles.Contains(0))
        {
            return 1f;
        }

        return Enumerable.Range(0, tiles.Rank).Select(i => ((int)tiles[i].FixedValue).Ranges(0, (int)shape[i].FixedValue)).CartesianProduct().Select(rgs =>
        {
            var slice = rgs.ToArray();
            var iscontiguous = TensorUtilities.IsContiguousSlice(shape.ToValueArray(), slice, out var contiguousStart);
            var size = TensorUtilities.GetProduct(tiles.ToValueArray(), contiguousStart) * distributedType.TensorType.DType.SizeInBytes;
            var (div, rem) = Math.DivRem(size, burstLength);
            return ((div * 1.0f) + ((float)rem / burstLength)) / (div + 1);
        }).Average();
    }

    public static TensorType GetDividedTensorType(DistributedType distributedType)
    {
        var (tiles, _) = GetDividedTile(distributedType);
        return distributedType.TensorType with { Shape = tiles };
    }

    public static int[] GetUnraveledIndex(int index, int[] hierarchies)
    {
        var strides = TensorUtilities.GetStrides(hierarchies);
        int remain = index;
        var unraveledIndex = new int[hierarchies.Length];
        for (int i = 0; i < unraveledIndex.Length; i++)
        {
            unraveledIndex[i] = remain / strides[i];
            remain = remain % strides[i];
        }

        return unraveledIndex;
    }

    public static (long[] Offset, long[] Shape) GetLocalOffsetAndShape(DistributedType distributedType, int[] shardIndex)
    {
        var globalShape = distributedType.TensorType.Shape.ToValueArray();
        var offset = new long[distributedType.TensorType.Shape.Rank];
        var shape = new long[distributedType.TensorType.Shape.Rank];
        for (int axis = 0; axis < offset.Length; axis++)
        {
            var splits = (from d in distributedType.NdSBP.Select((s, i) => (s, i))
                          let s = d.s as SBPSplit
                          where s != null && s.Axis == axis
                          select (Placement: d.i, DeviceIndex: shardIndex[d.i], DeviceDim: distributedType.Placement.Hierarchy[d.i])).ToArray();
            if (splits.Any())
            {
                var subHierarchies = splits.Select(x => x.DeviceDim).ToArray();
                var subHierarchyStrides = TensorUtilities.GetStrides(subHierarchies);
                var subHierarchySize = (int)TensorUtilities.GetProduct(subHierarchies);
                var subShardIndex = splits.Select(x => x.DeviceIndex).ToArray();
                var linearIndex = TensorUtilities.GetIndex(subHierarchyStrides, subShardIndex);
                var localDim = MathUtility.CeilDiv(globalShape[axis], subHierarchySize);
                offset[axis] = linearIndex * localDim;
                shape[axis] = Math.Min(localDim, globalShape[axis] - offset[axis]);
            }
            else
            {
                offset[axis] = 0;
                shape[axis] = globalShape[axis];
            }
        }

        return (offset, shape);
    }

    private static (Shape Tile, Shape Shape) GetDividedTile(DistributedType distributedType)
    {
        var shape = distributedType.TensorType.Shape.ToArray();
        var tiles = distributedType.TensorType.Shape.ToArray();
        foreach (var (s, i) in distributedType.NdSBP.Select((s, i) => (s, i)).Where(t => t.s is SBPSplit).Select(t => ((SBPSplit)t.s, t.i)))
        {
            tiles[s.Axis] /= distributedType.Placement.Hierarchy[i];
        }

        return (tiles, shape);
    }
}
