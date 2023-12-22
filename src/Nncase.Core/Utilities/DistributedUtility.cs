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
            for (int axis = 0; axis < tensorType.Shape.Rank; axis++)
            {
                if (tensorType.Shape[axis] is { IsFixed: true, Value: int s } && IsDivideBy(s, placement.Hierarchy[i]))
                {
                    ndsbp.Add(SBP.S(axis));
                }
            }

            ndsbp.Add(SBP.B);
            ndsbps.Add(ndsbp);
        }

        return ndsbps.CartesianProduct().Select(ndsbp => ndsbp.ToArray()).Where(ndsbp => IsDistributable(tensorType, ndsbp, placement)).Select(ndsbp => new IRArray<SBP>(ndsbp)).ToArray();
    }

    public static IReadOnlyList<IRArray<SBP>> GetPartialCandidateNDSBPs(DistributedType distributedType)
    {
        IRArray<SBP> ndsbp = distributedType.NdSBP;
        TensorType tensorType = distributedType.TensorType;
        Placement placement = distributedType.Placement;
        if (!ndsbp.Any(sbp => sbp is SBPPartialSum))
        {
            return Array.Empty<IRArray<SBP>>();
        }

        var candidateNdsbps = new List<SBP>[placement.Rank];
        for (int i = 0; i < placement.Rank; i++)
        {
            candidateNdsbps[i] = new List<SBP>();
            var innerSplitedAxes = distributedType.NdSBP.Skip(i + 1).OfType<SBPSplit>().Select(sbp => sbp.Axis).ToList();
            if (ndsbp[i] is SBPPartialSum)
            {
                candidateNdsbps[i].Add(SBP.B);
                for (int axis = 0; axis < tensorType.Shape.Rank; axis++)
                {
                    if (tensorType.Shape[axis] is { IsFixed: true, Value: int s } && IsDivideBy(s, placement.Hierarchy[i]) && !innerSplitedAxes.Contains(axis))
                    {
                        candidateNdsbps[i].Add(SBP.S(axis));
                    }
                }
            }
            else
            {
                candidateNdsbps[i].Add(ndsbp[i]);
            }
        }

        return candidateNdsbps.CartesianProduct().Select(ndsbp => ndsbp.ToArray()).Where(ndsbp => IsDistributable(tensorType, ndsbp, placement)).Select(ndsbp => new IRArray<SBP>(ndsbp)).ToArray();
    }

    public static bool IsDistributable(TensorType tensorType, ReadOnlySpan<SBP> ndsbp, Placement placement)
    {
        if (!tensorType.Shape.IsFixed)
        {
            return false;
        }

        var divisors = GetDivisors(new DistributedType(tensorType, new IRArray<SBP>(ndsbp.ToArray()), placement));
        return divisors.Select((d, axis) => (d, axis)).All(p => p.d == 0 ? true : IsDivideBy(tensorType.Shape[p.axis].FixedValue, p.d));
    }

    public static IReadOnlyList<int> GetDivisors(DistributedType distributedType)
    {
        var shape = distributedType.TensorType.Shape.ToValueArray();
        var divisors = Enumerable.Repeat(0, shape.Length).ToArray();
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
        if (divisors.Select((d, i) => (d, i)).All(p => p.d == 0 || IsDivideExactly(distributedType.TensorType.Shape[p.i].FixedValue, p.d)))
        {
            tensorType = new TensorType(distributedType.TensorType.DType, distributedType.TensorType.Shape.Zip(divisors).Select(p => p.Second == 0 ? p.First.FixedValue : p.First.FixedValue / p.Second).ToArray());
            return true;
        }

        return false;
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

    public static List<int[]> TryGetNonUniformDividedSlice(DistributedType distributedType)
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

        var spliList = hierarchies.Select<List<int>, int[]>((divs, axis) =>
        {
            int[] dim;
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

        IEnumerable<int[]> ret = new[] { Array.Empty<int>() };
        foreach (int[] array in spliList)
        {
            ret = from seq in ret
                  from item in array
                  select seq.Concat(new[] { item }).ToArray();
        }

        return ret.ToList();
    }

    public static bool IsDivideBy(int input, int divisor)
    {
        if (input >= divisor)
        {
            return true;
        }

        return false;
    }

    public static bool IsDivideExactly(int input, int divisor)
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
        return Enumerable.Range(0, tiles.Count).Select(i => tiles[i].Ranges(0, shape[i])).CartesianProduct().Select(rgs =>
        {
            var slice = rgs.ToArray();
            var iscontiguous = TensorUtilities.IsContiguousSlice(shape.ToArray(), slice, out var contiguousStart);
            var size = TensorUtilities.GetProduct(tiles.ToArray(), contiguousStart) * distributedType.TensorType.DType.SizeInBytes;
            var (div, rem) = Math.DivRem(size, burstLength);
            return ((div * 1.0f) + ((float)rem / burstLength)) / (div + 1);
        }).Average();
    }

    public static TensorType GetDividedTensorType(DistributedType distributedType)
    {
        var (tiles, _) = GetDividedTile(distributedType);
        return distributedType.TensorType with { Shape = new Shape(tiles) };
    }

    private static (IReadOnlyList<int> Tile, IReadOnlyList<int> Shape) GetDividedTile(DistributedType distributedType)
    {
        var shape = distributedType.TensorType.Shape.ToValueArray();
        var tiles = distributedType.TensorType.Shape.ToValueArray();
        foreach (var (s, i) in distributedType.NdSBP.Select((s, i) => (s, i)).Where(t => t.s is SBPSplit).Select(t => ((SBPSplit)t.s, t.i)))
        {
            tiles[s.Axis] /= distributedType.Placement.Hierarchy[i];
        }

        return (tiles, shape);
    }
}
