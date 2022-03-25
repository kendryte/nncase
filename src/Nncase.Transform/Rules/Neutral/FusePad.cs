// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Numerics.Tensors;
using Nncase.IR;
using Nncase.IR.NN;
using Nncase.PatternMatch;
using static Nncase.IR.F.NN;
using static Nncase.IR.F.Tensors;
using static Nncase.PatternMatch.F.NN;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Transform.Rules.Neutral;

/// <summary>
/// Fuse <see cref="IR.NN.Pad"/> into <see cref="IR.NN.Conv2D"/>.
/// </summary>
[RuleGenerator]
public sealed partial class FusePadConv2d : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } = IsConv2D(
        PadMode.Constant,
        IsPad(PadMode.Constant, IsWildcard("input"), IsWildcard("pads1"), 0),
        IsWildcard("weights"),
        IsWildcard("bias"),
        IsWildcard("stride"),
        IsWildcard("pads2"),
        IsWildcard("dilation"),
        IsWildcard("groups"));

    private Expr? GetReplace(Expr input, Expr pads1, Expr weights, Expr bias, Expr stride, Expr pads2, Expr dilation, Expr groups)
    {
        // todo:need fix
        var newPadsH = new int[] { 0, 0 };
        var newPadsW = new int[] { 0, 0 };
        var convPadsH = Stack(new IR.Tuple(pads1[2] + pads2[0], pads1[6] + pads2[2]), 0);
        var convPadsW = Stack(new IR.Tuple(pads1[3] + pads2[1], pads1[7] + pads2[3]), 0);
        var newPadsN = Stack(new IR.Tuple(pads1[0], pads1[4]), 0);
        var newPadsC = Stack(new IR.Tuple(pads1[1], pads1[5]), 0);

        var newPads = Concat(new IR.Tuple(newPadsN, newPadsC, newPadsH, newPadsW), 0);
        var convPads = Concat(new IR.Tuple(convPadsH, convPadsW), 0);

        return Conv2D(Pad(input, newPads, PadMode.Constant, 0), weights, bias, stride, convPads, dilation, PadMode.Constant, groups);
    }
}
