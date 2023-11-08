// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
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
                if (tensorType.Shape[axis] is { IsFixed: true, Value: int s } && IsDivisible(s, placement.Hierarchy[i]))
                {
                    ndsbp.Add(SBP.S(axis));
                }
            }

            ndsbp.Add(SBP.B);
            ndsbps.Add(ndsbp);
        }

        return ndsbps.CartesianProduct().
           Select(ndsbp => ndsbp.ToArray()).
           Where(ndsbp => IsDistributable(tensorType, ndsbp, placement, out _)).
           Select(ndsbp => new IRArray<SBP>(ndsbp)).
           ToArray();
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
                    if (tensorType.Shape[axis] is { IsFixed: true, Value: int s } && IsDivisible(s, placement.Hierarchy[i]) && !innerSplitedAxes.Contains(axis))
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

        return candidateNdsbps.CartesianProduct().
            Select(ndsbp => ndsbp.ToArray()).
            Where(ndsbp => IsDistributable(tensorType, ndsbp, placement, out _)).
            Select(ndsbp => new IRArray<SBP>(ndsbp)).
            ToArray();
    }

    public static bool IsDistributable(TensorType tensorType, ReadOnlySpan<SBP> ndsbp, Placement placement, [MaybeNullWhen(false)] out TensorType distType)
    {
        distType = null;
        if (!tensorType.Shape.IsFixed)
        {
            return false;
        }

        var shape = tensorType.Shape.ToValueArray();
        for (int i = 0; i < ndsbp.Length; i++)
        {
            if (ndsbp[i] is SBPSplit { Axis: int axis })
            {
                if (!IsDivisible(shape[axis], placement.Hierarchy[i]))
                {
                    return false;
                }

                shape[axis] /= placement.Hierarchy[i];
            }
        }

        distType = tensorType with { Shape = shape };
        return true;
    }

    public static bool IsDivisible(int input, int divisor)
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
        return Enumerable.Range(0, tiles.Count).
                  Select(i => tiles[i].Ranges(0, shape[i])).
                  CartesianProduct().
                  Select(rgs =>
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
