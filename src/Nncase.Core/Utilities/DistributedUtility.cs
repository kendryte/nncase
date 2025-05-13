// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using NetFabric.Hyperlinq;
using Nncase.IR;

namespace Nncase.Utilities;

public static class DistributedUtility
{
    public static List<List<int>> GetHierarchyCombinations(int rank)
    {
        var allCombinations = new List<List<int>>(rank);
        for (int length = 1; length <= rank; length++)
        {
            GetCombinations(Enumerable.Range(0, rank).ToArray(), length, 0, new List<int>(), allCombinations);
        }

        return allCombinations;
    }

    public static void GetCombinations(int[] array, int length, int startIndex, List<int> current, List<List<int>> result)
    {
        if (current.Count == length)
        {
            result.Add([.. current]);
            return;
        }

        for (int i = startIndex; i < array.Length; i++)
        {
            current.Add(array[i]);
            GetCombinations(array, length, i + 1, current, result);
            current.RemoveAt(current.Count - 1);
        }
    }

    public static IReadOnlyList<IRArray<SBP>> GetLeafCandidatePolicies(TensorType tensorType, Placement placement)
    {
        var splitsAxes = GetHierarchyCombinations(placement.Rank);
        var policies = new List<List<SBP>>();
        for (int di = 0; di < tensorType.Shape.Rank; di++)
        {
            var policy = new List<SBP>();
            if (tensorType.Shape.All(x => x.IsDynamic || x.FixedValue != 0))
            {
                for (int ti = 0; ti < splitsAxes.Count; ti++)
                {
                    var axis = splitsAxes[ti];
                    var divisor = axis.Select(a => placement.Hierarchy[a]).Aggregate(1, (a, b) => a * b);
                    if (tensorType.Shape[di] is { IsFixed: true, FixedValue: long s } && axis.All(a => placement.Hierarchy[a] > 1) && divisor > 1 && IsDivideExactly(s, divisor))
                    {
                        policy.Add(SBP.S(axis.ToArray()));
                    }
                }
            }

            policy.Add(SBP.B);
            policies.Add(policy);
        }

        var candidates = policies.CartesianProduct().Select(policy => policy.ToArray()).Where(policy => IsDistributable(tensorType, policy, placement)).Select(policy => new IRArray<SBP>(policy)).ToArray();
        return candidates;
    }

    public static IReadOnlyList<IRArray<SBP>> GetPartialCandidateNDSBPs(DistributedType distributedType)
    {
        IRArray<SBP> ndsbp = distributedType.AxisPolices;
        TensorType tensorType = distributedType.TensorType;
        Placement placement = distributedType.Placement;
        if (!ndsbp.Any(sbp => sbp is SBPPartial))
        {
            return Array.Empty<IRArray<SBP>>();
        }

        var candidateNdsbps = new List<SBP>[placement.Rank];
        for (int i = 0; i < placement.Rank; i++)
        {
            candidateNdsbps[i] = new List<SBP>();

            // var innerSplitedAxes = distributedType.NdSBP.Skip(i + 1).OfType<SBPSplit>().Select(sbp => sbp.Axis).ToList();
            if (ndsbp[i] is SBPPartial)
            {
                candidateNdsbps[i].Add(SBP.B);

                // note separate reduce boxing and reshard boxing.
                // for (int axis = 0; axis < tensorType.Shape.Rank; axis++)
                // {
                //     if (tensorType.Shape[axis] is { IsFixed: true, Value: int s } && placement.Hierarchy[i] > 1 && IsDivideExactly(s, placement.Hierarchy[i]) && !innerSplitedAxes.Contains(axis))
                //     {
                //         candidateNdsbps[i].Add(SBP.S(axis));
                //     }
                // }
            }
            else
            {
                candidateNdsbps[i].Add(ndsbp[i]);
            }
        }

        return candidateNdsbps.CartesianProduct().Select(ndsbp => ndsbp.ToArray()).Where(ndsbp => IsDistributable(tensorType, ndsbp, placement)).Select(ndsbp => new IRArray<SBP>(ndsbp)).ToArray();
    }

    public static bool IsDistributable(TensorType tensorType, ReadOnlySpan<SBP> polices, Placement placement)
    {
        if (!tensorType.Shape.IsRanked)
        {
            return false;
        }

        // 1. S on different dim must have different topology axis.
        if (!IsDistributable(polices))
        {
            return false;
        }

        // 2. All shapes are divisible by the mesh.
        var divisors = GetDivisors(new DistributedType(tensorType, polices.ToArray(), placement));
        return divisors.Select((d, axis) => (d, axis)).All(p => p.d == 0 ? true : IsDivideExactly(tensorType.Shape[p.axis].FixedValue, p.d));
    }

    public static bool IsDistributable(ReadOnlySpan<SBP> polices)
    {
        var splits = polices.ToArray().Where(p => p is SBPSplit).Select(p => (SBPSplit)p).ToArray();
        if (splits == null || splits.Length == 0 || (splits.Length < 2 && splits[0].Axes.GroupBy(x => x).All(group => group.Count() == 1)))
        {
            return true;
        }

        for (int i = 0; i < splits.Length - 1; i++)
        {
            for (int j = i + 1; j < splits.Length; j++)
            {
                if (splits[i].Axes.Intersect(splits[j].Axes).Any())
                {
                    return false;
                }
            }
        }

        return true;
    }

    public static IReadOnlyList<int> GetDivisors(DistributedType distributedType)
    {
        var rank = distributedType.TensorType.Shape.Rank;
        var divisors = Enumerable.Repeat(0, rank).ToArray();
        for (int i = 0; i < distributedType.AxisPolices.Count; i++)
        {
            if (distributedType.AxisPolices[i] is SBPSplit split)
            {
                foreach (var a in split.Axes)
                {
                    if (divisors[i] == 0)
                    {
                        divisors[i] = 1;
                    }

                    divisors[i] *= distributedType.Placement.Hierarchy[a];
                }
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

    public static IRArray<SBP> AxisPolicesToNDSBP(IRArray<SBP> axisPolices, int rank)
    {
        var ndsbp = new SBP[rank];
        for (var i = 0; i < axisPolices.Count; i++)
        {
            var policy = axisPolices[i];
            if (policy is SBPSplit split)
            {
                foreach (var ax in split.Axes)
                {
                    ndsbp[ax] = SBP.S(new[] { i });
                }
            }
        }

        return ndsbp.Select(sbp => sbp is SBPSplit ? sbp : SBP.B).ToArray();
    }

    public static IRArray<SBP> NDSBPToAxisPolices(IRArray<SBP> ndsbp, int rank)
    {
        var polices = new SBP[rank];
        for (int d = 0; d < polices.Length; d++)
        {
            var splitAxes = Enumerable.Range(0, ndsbp.Count).Where(i => ndsbp[i] is SBPSplit split && split.Axes[0] == d).ToArray();
            if (splitAxes.Any())
            {
                polices[d] = SBP.S(splitAxes);
            }
            else
            {
                polices[d] = SBP.B;
            }
        }

        return polices;
    }

#if false
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
#endif

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
            var splits = distributedType.AxisPolices[axis] is SBPSplit s
            ? s.Axes.Select(td => (Placement: td, DeviceIndex: shardIndex[td], DeviceDim: distributedType.Placement.Hierarchy[td])).ToArray()
            : Array.Empty<(int Placement, int DeviceIndex, int DeviceDim)>();
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
        for (var d = 0; d < shape.Length; d++)
        {
            if (distributedType.AxisPolices.Count > d && distributedType.AxisPolices[d] is SBPSplit split)
            {
                tiles[d] /= split.Axes.Select(t => distributedType.Placement.Hierarchy[t]).Aggregate(1, (a, b) => a * b);
            }
        }

        return (tiles, shape);
    }
}
