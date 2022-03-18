// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Linq;
using System.Numerics.Tensors;
using Nncase.IR;
using Nncase.PatternMatch;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Transform.Rules.Neutral;

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
                || ends[i] != inShape[axis]
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

    private bool IsNoSlice(Shape inShape, Tensor<int> begins, Tensor<int> ends, Tensor<int> axes, Tensor<int> strides, int dim)
    {
        return Enumerable.Range(0, begins.Length)
            .All(i => begins[i] == 0 && ends[i] == inShape[i].FixedValue && strides[i] == 1);
    }

    private Expr? GetReplace(Expr input, Tensor<int> begins1, Tensor<int> ends1, Tensor<int> axes1, Tensor<int> strides1, Tensor<int> begins2, Tensor<int> ends2, Tensor<int> axes2, Tensor<int> strides2)
    {
        var inShape = input.CheckedShape;

        bool CanMerge()
        {
            return Enumerable.Range(0, inShape.Rank).All(
              dim => (IsNoSlice(inShape, begins1, ends1, axes1, strides1, dim)
                || IsNoSlice(inShape, begins2, ends2, axes2, strides2, dim))
                && axes1[dim] == axes2[dim]);
        }

        if (!CanMerge())
        {
            return null;
        }

        var new_begins = new Tensor<int>(begins1.Dimensions);
        var new_ends = new Tensor<int>(ends1.Dimensions);
        var new_strides = new Tensor<int>(strides1.Dimensions);
        var rank = inShape.Rank;
        for (int dim = 0; dim < rank; dim++)
        {
            var isNoSlice1 = IsNoSlice(inShape, begins1, ends1, axes1, strides1, dim);

            new_begins[dim] = isNoSlice1 ? begins2[dim] : begins1[dim];
            new_ends[dim] = isNoSlice1 ? ends2[dim] : ends1[dim];
            new_strides[dim] = isNoSlice1 ? strides2[dim] : strides1[dim];
        }

        return Slice(input, new_begins, new_ends, axes1, new_strides);
    }
}
