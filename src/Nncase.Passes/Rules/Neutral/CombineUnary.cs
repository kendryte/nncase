// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.NN;
using Nncase.IR.Tensors;
using Nncase.PatternMatch;
using static Nncase.IR.F.Math;
using static Nncase.IR.F.NN;
using static Nncase.IR.F.Tensors;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.F.NN;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Neutral;

/// <summary>
/// Combine <see cref="IR.Math.Unary"/>(<see cref="IR.NN.Pad"/>).
/// </summary>
[RuleGenerator]
public sealed partial class CombinePadUnary : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } = IsUnary(
        "unary",
        x => true,
        IsPad(
            "pad",
            x => true,
            IsWildcard("input"),
            IsWildcard("pads"),
            IsWildcard("padValue")));

    private Expr? GetReplace(Unary unary, Pad pad, Expr input, Expr pads, Expr padValue)
    {
        if (pad.PadMode == PadMode.Constant)
        {
            var newPadValue = Unary(unary.UnaryOp, padValue);
            return Pad(Unary(unary.UnaryOp, input), pads, PadMode.Constant, newPadValue);
        }
        else
        {
            return Pad(Unary(unary.UnaryOp, input), pads, pad.PadMode, 0f);
        }
    }
}

/// <summary>
/// Combine <see cref="IR.Math.Unary"/>(<see cref="IR.Tensors.Transpose"/>).
/// </summary>
[RuleGenerator]
public sealed partial class CombineTranposeUnary : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } = IsUnary(
        "unary",
        x => true,
        IsTranspose(IsWildcard("input"), IsWildcard("perm")));

    private Expr? GetReplace(Unary unary, Expr input, Expr perm)
    {
        return Transpose(Unary(unary.UnaryOp, input), perm);
    }
}

/// <summary>
/// Combine <see cref="IR.Math.Unary"/>(<see cref="IR.Tensors.Slice"/>).
/// </summary>
[RuleGenerator]
public sealed partial class CombineSliceUnary : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } = IsUnary(
        "unary",
        x => true,
        IsSlice(IsWildcard("input"), IsWildcard("begins"), IsWildcard("ends"), IsWildcard("axes"), IsWildcard("strides")));

    private Expr? GetReplace(Unary unary, Expr input, Expr begins, Expr ends, Expr axes, Expr strides)
    {
        return Slice(Unary(unary.UnaryOp, input), begins, ends, axes, strides);
    }
}

/// <summary>
/// Combine <see cref="IR.Math.Unary"/>(<see cref="IR.Tensors.Reshape"/>).
/// </summary>
[RuleGenerator]
public sealed partial class CombineReshapeUnary : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } = IsUnary(
        "unary",
        x => true,
        IsReshape(IsWildcard("input"), IsWildcard("shape")));

    private Expr? GetReplace(Unary unary, Expr input, Expr shape)
    {
        return Reshape(Unary(unary.UnaryOp, input), shape);
    }
}
