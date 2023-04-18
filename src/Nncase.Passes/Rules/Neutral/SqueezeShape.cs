// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.NN;
using Nncase.IR.Tensors;
using Nncase.PatternMatch;
using Tensorflow;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;
using Tuple = Nncase.IR.Tuple;

namespace Nncase.Passes.Rules.Neutral;

/// <summary>
/// Squeeze transpose shape.
/// </summary>
[RuleGenerator]
public sealed partial class SqueezeTransposeShape : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } = IsTranspose(IsWildcard("input") with { TypePattern = HasFixedShape() & HasRank(x => x > 4, "more than 4D need to squeeze") }, IsWildcard("perm"));

    private Tuple<bool, List<int>, List<int>> SqueezeTranspose(List<int> oldShape, List<int> oldAxis)
    {
        if (oldShape.Count <= 4)
        {
            return new Tuple<bool, List<int>, List<int>>(false, oldAxis, oldShape);
        }

        var newAxis = new List<int>(oldAxis);
        var newShape = new List<int>(oldShape);
        int squeezeTimes = oldShape.Count - 4;

        var foldIndexCouple = new List<Tuple<int, int>>();
        for (int i = oldShape.Count - 1; i > 0; i--)
        {
            if (oldAxis[i - 1] + 1 == oldAxis[i])
            {
                foldIndexCouple.Add(new Tuple<int, int>(i - 1, i));
            }
        }

        if (foldIndexCouple.Count < squeezeTimes)
        {
            return new Tuple<bool, List<int>, List<int>>(false, newAxis, newShape);
        }

        while (squeezeTimes > 0 && foldIndexCouple.Count > 0)
        {
            Tuple<int, int> it = foldIndexCouple[0];
            int front = it.Item1;
            int back = it.Item2;
            newShape[oldAxis[front]] *= newShape[oldAxis[back]];
            newShape.RemoveAt(oldAxis[back]);
            newAxis.RemoveAt(back);
            foldIndexCouple.RemoveAt(0);
            squeezeTimes--;
        }

        // fix axis
        for (int i = 0, j = 0; j < 4; i++)
        {
            int index = newAxis.IndexOf(i);
            if (index != -1)
            {
                newAxis[index] = j;
                j++;
            }
        }

        return new Tuple<bool, List<int>, List<int>>(true, newAxis, newShape);
    }

    private Expr? GetReplace(Expr input, int[] perm)
    {
        var inputShape = input.CheckedShape;
        var (result, new_perm, new_shape) = SqueezeTranspose(inputShape.ToValueList(), perm.ToList());
        if (!result)
        {
            return null;
        }

        var newOutputShape = new int[perm.Length];
        for (int i = 0; i < perm.Length; i++)
        {
            newOutputShape[i] = inputShape[perm[i]].FixedValue;
        }

        return Reshape(Transpose(Reshape(input, new_shape.ToArray()), new_perm.ToArray()), newOutputShape);
    }
}
