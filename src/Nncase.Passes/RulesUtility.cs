// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Linq;

namespace Nncase.Passes;

public static class RulesUtility
{
    /// <summary>
    /// find sequeezed axis index.
    /// </summary>
    /// <param name="oldShape">old shape.</param>
    /// <param name="newShape">new shape.</param>
    /// <returns>axis, if not found return -1.</returns>
    public static int FindSqueezeAxis(int[] oldShape, int[] newShape)
    {
        if (oldShape.Length <= newShape.Length)
        {
            return -1;
        }

        var indices = Enumerable.Range(0, oldShape.Length).ToList();
        foreach (var dim in newShape)
        {
            for (int i = 0; i < oldShape.Length; i++)
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
}
