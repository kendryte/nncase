// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Nncase.IR;
using Nncase.IR.NN;
using Nncase.PatternMatch;
using static Nncase.IR.F.NN;
using static Nncase.IR.F.Tensors;
using static Nncase.PatternMatch.F.NN;
using static Nncase.PatternMatch.Utility;
using Random = Nncase.IR.F.Random;

namespace Nncase.Passes.Rules.Neutral;

/// <summary>
/// Fuse <see cref="IR.NN.Pad"/> into <see cref="IR.NN.Conv2D"/>.
/// </summary>
[RuleGenerator]
public sealed partial class FusePadConv2d : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } = IsConv2D(
        PadMode.Constant,
        IsPad(
            PadMode.Constant,
            IsWildcard("input"),
            IsWildcard("pads1"),
            IsWildcard("value")),
        IsWildcard("weights"),
        IsWildcard("bias"),
        IsWildcard("stride"),
        IsWildcard("pads2"),
        IsWildcard("dilation"),
        IsWildcard("groups"),
        IsWildcard("fusedClamp"));

    private Expr? GetReplace(Expr input, Expr pads1, Expr weights, Expr bias, Expr stride, Expr pads2, Expr dilation, Expr groups, Expr fusedClamp)
    {
        var newPadsH = new[] { 0, 0 };
        var newPadsW = new[] { 0, 0 };

        var needPaddingShape = pads1.Evaluate().AsTensor();
        if (needPaddingShape[2, 0] is 0
            && needPaddingShape[2, 1] is 0
            && needPaddingShape[3, 0] is 0
            && needPaddingShape[3, 1] is 0)
        {
            return null;
        }

        var convPadsH = Stack(new IR.Tuple(pads1[2, 0] + pads2[0, 0], pads1[2, 1] + pads2[0, 1]), 0);
        var convPadsW = Stack(new IR.Tuple(pads1[3, 0] + pads2[1, 0], pads1[3, 1] + pads2[1, 1]), 0);
        var newPads = Stack(new IR.Tuple(pads1[0], pads1[1], newPadsH, newPadsW), 0);
        var convPads = Stack(new IR.Tuple(convPadsH, convPadsW), 0);

        return Conv2D(Pad(input, newPads, PadMode.Constant, 0f), weights, bias, stride, convPads, dilation, PadMode.Constant, groups, fusedClamp);
    }
}
