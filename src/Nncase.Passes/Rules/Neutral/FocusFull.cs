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

[RuleGenerator]
public sealed partial class FocusFull : RewriteRule<Pattern>
{
    private static readonly Pattern Input = IsWildcard("input") with { TypePattern = HasRank(4) };

    /// <inheritdoc />
    public override Pattern Pattern { get; } = IsConcat(
      "concat",
      "concatCall",
      _ => true,
      PatternMatch.Utility.IsTuple("tp", new[] {
        IsSlice(Input, IsTensorConst("begin0"), IsTensorConst("end0"), IsTensorConst("axes0"), IsTensorConst("stride0")),
        IsSlice(Input, IsTensorConst("begin1"), IsTensorConst("end1"), IsTensorConst("axes1"), IsTensorConst("stride1")),
        IsSlice(Input, IsTensorConst("begin2"), IsTensorConst("end2"), IsTensorConst("axes2"), IsTensorConst("stride2")),
        IsSlice(Input, IsTensorConst("begin3"), IsTensorConst("end3"), IsTensorConst("axes3"), IsTensorConst("stride3")),
      }));

    private Expr? GetReplace(IR.Tensors.Concat concat, Call concatCall, Expr input, int[] begin0, long[] end0, int[] axes0, int[] stride0, int[] begin1, long[] end1, int[] axes1, int[] stride1, int[] begin2, long[] end2, int[] axes2, int[] stride2, int[] begin3, long[] end3, int[] axes3, int[] stride3)
    {
        int axis = concat.Axis;
        var inputShape = input.CheckedShape.ToValueArray();
        if (inputShape[0] != 1)
        {
            return null;
        }

        if (axis != 1)
        {
            return null;
        }

        // note only support nchw
        var axes = new[] { 2, 3 };
        if (!Enumerable.SequenceEqual(axes0, axes) ||
            !Enumerable.SequenceEqual(axes1, axes) ||
            !Enumerable.SequenceEqual(axes2, axes) ||
            !Enumerable.SequenceEqual(axes3, axes))
        {
            return null;
        }

        var begins = new[] { begin0, begin1, begin2, begin3 };
        var ends = new[] { end0, end1, end2, end3 };
        var strides = new[] { stride0, stride1, stride2, stride3 };

        if (!ends.All(e => e[0] >= inputShape[axes[0]]) || !ends.All(e => e[1] >= inputShape[axes[1]]) ||
            !Enumerable.SequenceEqual(strides.Select(s => s[0]), new[] { 2, 2, 2, 2 }) ||
            !Enumerable.SequenceEqual(strides.Select(s => s[1]), new[] { 2, 2, 2, 2 }))
        {
            return null;
        }

        if (Enumerable.SequenceEqual(begins.Select(b => b[0]), new[] { 0, 1, 0, 1 }) &&
           Enumerable.SequenceEqual(begins.Select(b => b[1]), new[] { 0, 0, 1, 1 }))
        {
            var b1 = IR.F.Tensors.Reshape(input, new[] { inputShape[1], inputShape[2] / 2, 2, inputShape[3] });
            var t1 = Transpose(b1, new[] { 2, 0, 1, 3 });
            var b2 = IR.F.Tensors.Reshape(t1, new[] { inputShape[1] * 2, inputShape[2] / 2, inputShape[3] / 2, 2 });
            var t2 = Transpose(b2, new[] { 3, 0, 1, 2 });
            var b3 = IR.F.Tensors.Reshape(t2, new[] { 1, inputShape[1] * 2 * 2, inputShape[2] / 2, inputShape[3] / 2 });
            return b3;
        }

        return null;
    }
}
