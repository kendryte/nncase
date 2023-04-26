// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.IR;
using Nncase.PatternMatch;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Neutral;

/// <summary>
/// Fold nop <see cref="IR.Tensors.Slice"/>.
/// </summary>
[RuleGenerator]
public sealed partial class FoldNopSlice : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } = IsSlice(
        IsWildcard("input") with { TypePattern = HasFixedShape() },
        IsTensorConst("begins"),
        IsTensorConst("ends"),
        IsTensorConst("axes"),
        IsTensorConst("strides"));

    private Expr? GetReplace(Expr input, Tensor<int> begins, Tensor<int> ends, Tensor<int> axes, Tensor<int> strides)
    {
        var inShape = input.CheckedShape;
        for (int i = 0; i < axes.Length; i++)
        {
            var axis = axes[i];
            if (begins[i] != 0
                || ends[i] != inShape[axis].FixedValue
                || strides[i] != 1)
            {
                return null;
            }
        }

        return input;
    }
}

/// <summary>
/// Fold two <see cref="IR.Tensors.Slice"/>.
/// </summary>
[RuleGenerator]
public sealed partial class FoldTwoSlices : RewriteRule<Pattern>
{
    /// <inheritdoc/>
    public override Pattern Pattern { get; } = IsSlice(
        IsSlice(IsWildcard("input") with { TypePattern = HasFixedShape() }, IsTensorConst("begins1"), IsTensorConst("ends1"), IsTensorConst("axes1"), IsTensorConst("strides1")),
        IsTensorConst("begins2"),
        IsTensorConst("ends2"),
        IsTensorConst("axes2"),
        IsTensorConst("strides2"));

    private Expr? GetReplace(Expr input, int[] begins1, long[] ends1, int[] axes1, int[] strides1, int[] begins2, long[] ends2, int[] axes2, int[] strides2)
    {
        var inShape = input.CheckedShape;

        var newBegins = new List<int>();
        var newEnds = new List<long>();
        var newStrides = new List<int>();
        var newAxes = new List<int>();

        for (int axis = 0; axis < inShape.Rank; axis++)
        {
            int i1 = axes1.ToList().IndexOf(axis);
            int i2 = axes2.ToList().IndexOf(axis);

            if ((i1, i2) is (-1, -1))
            {
                continue;
            }
            else if ((i1, i2) is (_, -1) or (-1, _))
            {
                // apply slice on different slice.
                if (i1 != -1)
                {
                    newBegins.Add(begins1[i1]);
                    newEnds.Add(ends1[i1]);
                    newStrides.Add(strides1[i1]);
                    newAxes.Add(axes1[i1]);
                }
                else
                {
                    newBegins.Add(begins2[i2]);
                    newEnds.Add(ends2[i2]);
                    newStrides.Add(strides2[i2]);
                    newAxes.Add(axes2[i2]);
                }
            }
            else
            {
                // todo add same axis slice fold
                return null;
            }
        }

        return Slice(input, newBegins.ToArray(), newEnds.ToArray(), newAxes.ToArray(), newStrides.ToArray());
    }
}
