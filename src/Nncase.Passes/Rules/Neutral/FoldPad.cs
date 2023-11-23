// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.NN;
using Nncase.PatternMatch;
using static Nncase.IR.F.NN;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.NN;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Neutral;

/// <summary>
/// Fold nop <see cref="IR.NN.Pad"/>.
/// </summary>
[RuleGenerator]
public sealed partial class FoldNopPad : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } = IsPad(padMode => true, IsWildcard("input"), IsTensorConst("pads", IsIntegral()), IsWildcard());

    private Expr? GetReplace(Expr input, TensorConst pads)
    {
        if (pads.Value.Cast<int>().All(x => x == 0))
        {
            return input;
        }

        return null;
    }
}

/// <summary>
/// Fold two <see cref="IR.NN.Pad"/>.
/// </summary>
[RuleGenerator]
public sealed partial class FoldTwoPads : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } =
        IsPad(
            PadMode.Constant,
            IsPad(PadMode.Constant, IsWildcard("input"), IsTensorConst("pads1", IsIntegral()), IsWildcard("padValue1")),
            IsTensorConst("pads2", IsIntegral()),
            IsWildcard("padValue2"));

    private Expr? GetReplace(Expr input, TensorConst pads1, TensorConst pads2, Expr padValue1, Expr padValue2)
    {
        if (padValue1.Equals(padValue2))
        {
            var (t1, t2) = (pads1.Value.Cast<int>(), pads2.Value.Cast<int>());
            var newt = new Tensor<int>(t1.Dimensions);
            for (int i = 0; i < t1.Dimensions[0]; i++)
            {
                newt[i, 0] = t1[i, 0] + t2[i, 0];
                newt[i, 1] = t1[i, 1] + t2[i, 1];
            }

            return Pad(input, newt, PadMode.Constant, padValue1);
        }

        return null;
    }
}

/// <summary>
/// fold conv2d(pad(input)) => conv2d(input).
/// </summary>
[RuleGenerator]
public sealed partial class FoldConv2DPads : IRewriteRule
{
    public IPattern Pattern { get; } = IsConv2D(
        "conv",
        conv => conv.PadMode == PadMode.Constant,
        MaybeMarker(IsPad(
            pad => pad.PadMode == PadMode.Constant,
            IsWildcard("input"),
            IsTensorConst("ext_pad"),
            IsTensorConst("ext_pad_init"))),
        IsWildcard("weights"),
        IsWildcard("bias"),
        IsWildcard("stride"),
        IsTensorConst("padding"),
        IsWildcard("dilation"),
        IsWildcard("groups"),
        IsWildcard("fusedClamp"));

    private Expr? GetReplace(Conv2D conv, Expr input, Expr weights, Expr bias, Expr stride, Tensor<int> padding, Expr dilation, Expr groups, Expr fusedClamp, Tensor<int> ext_pad, float ext_pad_init)
    {
        if (!(ext_pad[0, 0] == 0 && ext_pad[0, 1] == 0 &&
              ext_pad[1, 0] == 0 && ext_pad[1, 1] == 0))
        {
            return null;
        }

        var new_pad = padding.Clone();
        new_pad[0, 0] += ext_pad[2, 0];
        new_pad[0, 1] += ext_pad[2, 1];
        new_pad[1, 0] += ext_pad[3, 0];
        new_pad[1, 1] += ext_pad[3, 1];
        return Conv2D(input, weights, bias, stride, new_pad, dilation, PadMode.Constant, groups, fusedClamp);
    }
}

/// <summary>
/// fold reduce_window(pad(input)) => reduce_window(input).
/// </summary>
[RuleGenerator]
public sealed partial class FoldReduceWindow2DPads : IRewriteRule
{
    public IPattern Pattern { get; } = IsReduceWindow2D(
        "pdp",
        _ => true,
        IsPad(
            pad => pad.PadMode == PadMode.Constant,
            IsWildcard("input"),
            IsTensorConst("ext_pad"),
            IsTensorConst("ext_pad_init")),
        IsWildcard("initValue"),
        IsWildcard("filter"),
        IsWildcard("stride"),
        IsTensorConst("padding"),
        IsWildcard("dilation"),
        IsWildcard("ceilMode"),
        IsWildcard("countIncludePad"));

    private Expr? GetReplace(ReduceWindow2D pdp, Expr input, Expr initValue, Expr filter, Expr stride, Tensor<int> padding, Expr dilation, Expr ceilMode, Expr countIncludePad, Tensor<int> ext_pad, float ext_pad_init)
    {
        if (!(ext_pad[0, 0] == 0 && ext_pad[0, 1] == 0 &&
              ext_pad[1, 0] == 0 && ext_pad[1, 1] == 0))
        {
            return null;
        }

        var new_pad = padding.Clone();
        new_pad[0, 0] += ext_pad[2, 0];
        new_pad[0, 1] += ext_pad[2, 1];
        new_pad[1, 0] += ext_pad[3, 0];
        new_pad[1, 1] += ext_pad[3, 1];
        return ReduceWindow2D(pdp.ReduceOp, input, initValue, filter, stride, new_pad, dilation, ceilMode, countIncludePad);
    }
}
