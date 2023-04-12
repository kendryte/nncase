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
public sealed partial class FoldTwoSlices : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } = IsSlice(
        IsSlice(IsWildcard("input") with { TypePattern = HasFixedShape() }, IsTensorConst("begins1"), IsTensorConst("ends1"), IsTensorConst("axes1"), IsTensorConst("strides1")),
        IsTensorConst("begins2"),
        IsTensorConst("ends2"),
        IsTensorConst("axes2"),
        IsTensorConst("strides2"));

    // private bool IsNoSlice(Shape inShape, Tensor<int> begins, Tensor<int> ends, Tensor<int> axes, Tensor<int> strides, int dim)
    // {
    //     return Enumerable.Range(0, begins.Length)
    //         .All(i => begins[i] == 0 && ends[i] == inShape[i].FixedValue && strides[i] == 1);
    // }
    private Expr? GetReplace(Expr input, Tensor<int> begins1, Tensor<int> ends1, Tensor<int> axes1, Tensor<int> strides1, Tensor<int> begins2, Tensor<int> ends2, Tensor<int> axes2, Tensor<int> strides2)
    {
        var inShape = input.CheckedShape;

        // bool CanMerge()
        // {
        //     return Enumerable.Range(0, inShape.Rank).All(
        //       dim => (IsNoSlice(inShape, begins1, ends1, axes1, strides1, dim)
        //         || IsNoSlice(inShape, begins2, ends2, axes2, strides2, dim))
        //         && axes1[dim] == axes2[dim]);
        // }

        // if (!CanMerge())
        // {
        //     return null;
        // }
        var newBegins = new List<int>();
        var newEnds = new List<int>();
        var newStrides = new List<int>();
        var newAxes = new List<int>();
        _ = inShape.Rank;

        for (int axis_1 = 0; axis_1 < axes1.Length; axis_1++)
        {
            var process_args = false;
            for (int axis_2 = 0; axis_2 < axes2.Length; axis_2++)
            {
                if (axes1[axis_1] == axes2[axis_2])
                {
                    newBegins.Add(strides1[axis_1] * begins2[axis_2]);
                    newEnds.Add(strides1[axis_1] * ends2[axis_2]);
                    newStrides.Add(strides1[axis_1] * strides2[axis_2]);
                    newAxes.Add(axes1[axis_1]);
                    process_args = true;
                }
                else
                {
                    newBegins.Add(begins2[axis_2]);
                    newEnds.Add(ends2[axis_2]);
                    newStrides.Add(strides2[axis_2]);
                    newAxes.Add(axes2[axis_2]);
                }
            }

            if (!process_args)
            {
                newBegins.Add(begins1[axis_1]);
                newEnds.Add(ends1[axis_1]);
                newStrides.Add(strides1[axis_1]);
                newAxes.Add(axes1[axis_1]);
            }
        }

        return Slice(input, newBegins.ToArray(), newEnds.ToArray(), newAxes.ToArray(), newStrides.ToArray());
    }
}
