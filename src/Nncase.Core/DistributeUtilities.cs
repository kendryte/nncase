// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using Nncase.IR;

namespace Nncase;

public static class DistributeUtilities
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
}
