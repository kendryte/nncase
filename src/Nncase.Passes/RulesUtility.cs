// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Linq;
using Nncase.IR;

namespace Nncase.Passes;

public static class RulesUtility
{
    /// <summary>
    /// find sequeezed axis index.
    /// </summary>
    /// <param name="oldShape">old shape.</param>
    /// <param name="newShape">new shape.</param>
    /// <returns>axis, if not found return -1.</returns>
    public static int FindSqueezeAxis(RankedShape oldShape, RankedShape newShape)
    {
        if (!IsSqueeze(oldShape, newShape) || oldShape.Rank <= newShape.Rank)
        {
            return -1;
        }

        var indices = Enumerable.Range(0, oldShape.Rank).ToList();
        foreach (var dim in newShape)
        {
            for (int i = 0; i < oldShape.Rank; i++)
            {
                if (oldShape[i] == dim && indices.IndexOf(i) != -1)
                {
                    indices.Remove(i);
                }
            }
        }

        var oneindex = (indices.Count == 1) ? indices[0] : -1;
        return oneindex;
    }

    /// <summary>
    /// if two shapes are squeeze/unsqueeze.
    /// </summary>
    /// <param name="oldShape">old shape.</param>
    /// <param name="newShape">new shape.</param>
    /// <returns>bool.</returns>
    public static bool IsSqueeze(RankedShape oldShape, RankedShape newShape)
    {
        var squeezedOldShape = oldShape.Where(x => x != Dimension.One).ToArray();
        var squeezedNewShape = newShape.Where(x => x != Dimension.One).ToArray();
        return squeezedOldShape.SequenceEqual(squeezedNewShape);
    }
}
