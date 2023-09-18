// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using Nncase.IR;

namespace Nncase;

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

    public static IReadOnlyList<Expr> GetLeafCandidateBoxings(Expr expr, Placement placement)
    {
        return GetLeafCandidateNDSBPs((TensorType)expr.CheckedType, placement).Select(ndsbp => IR.F.Tensors.Boxing(expr, new DistributedType((TensorType)expr.CheckedType, ndsbp, placement))).ToArray();
    }

    /// <summary>
    /// when input expression sbp is partial, get the new candidate boxings.
    /// </summary>
    /// <param name="expr">input expression.</param>
    /// <returns>the boxings.</returns>
    /// <exception cref="NotSupportedException">when expr is tuple.</exception>
    public static IReadOnlyList<Expr> GetPartialCandidateBoxings(Expr expr)
    {
        if (expr is IR.Tuple tuple)
        {
            var candidates = tuple.Fields.ToArray().
                Select(GetPartialCandidateBoxings).
                CartesianProduct();
            return candidates.Any() ? candidates.
                Select(fs => new IR.Tuple(fs.ToArray())).
                ToArray() : Array.Empty<Expr>();
        }

        var type = (DistributedType)expr.CheckedType;
        var tensorType = type.TensorType;
        var candidateNdsbps = new List<SBP>[type.Placement.Rank];
        for (int i = 0; i < type.Placement.Rank; i++)
        {
            candidateNdsbps[i] = new List<SBP>();
            if (type.NdSBP[i] is SBPPartialSum)
            {
                candidateNdsbps[i].Add(SBP.B);
                for (int axis = 0; axis < tensorType.Shape.Rank; axis++)
                {
                    if (tensorType.Shape[axis] is { IsFixed: true, Value: int s } && IsDivisible(s, type.Placement.Hierarchy[i]))
                    {
                        candidateNdsbps[i].Add(SBP.S(axis));
                    }
                }
            }
        }

        return candidateNdsbps.CartesianProduct().
            Select(ndsbp => new DistributedType(tensorType, new IRArray<SBP>(ndsbp), type.Placement)).
            Select(disttype => IR.F.Tensors.Boxing(expr, disttype)).ToArray();
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
        foreach (var (s, i) in distributedType.NdSBP.OfType<SBPSplit>().Select((s, i) => (s, i)))
        {
            tiles[s.Axis] /= distributedType.Placement.Hierarchy[i];
        }

        return (tiles, shape);
    }
}
