// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Nncase.IR;
using Nncase.IR.NN;
using Nncase.IR.Shapes;
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
            IsPaddings("pads1"),
            IsWildcard("value")),
        IsWildcard("weights"),
        IsWildcard("bias"),
        IsWildcard("stride"),
        IsPaddings("pads2"),
        IsWildcard("dilation"),
        IsWildcard("groups"),
        IsWildcard("fusedClamp"));

    private Expr? GetReplace(Expr input, Paddings pads1, Expr weights, Expr bias, Expr stride, Paddings pads2, Expr dilation, Expr groups, Expr fusedClamp)
    {
        var newPadsH = Padding.Zero;
        var newPadsW = Padding.Zero;

        if (pads1[2] == Padding.Zero
            && pads1[3] == Padding.Zero)
        {
            return null;
        }

        var convPadsH = pads1[2] + pads2[0];
        var convPadsW = pads1[3] + pads2[1];
        var newPads = new Paddings(pads1[0], pads1[1], newPadsH, newPadsW);
        var convPads = new Paddings(convPadsH, convPadsW);

        return Conv2D(Pad(input, newPads, PadMode.Constant, 0f), weights, bias, stride, convPads, dilation, PadMode.Constant, groups, fusedClamp);
    }
}
