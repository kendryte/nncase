// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.PatternMatch;
using static Nncase.IR.F.Math;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Neutral;

/// <summary>
/// Fold nop <see cref="IR.Math.Binary"/>.
/// </summary>
[RuleGenerator]
public sealed partial class FoldNopBinary : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } = IsBinary(
        "binary",
        x => x.BinaryOp is BinaryOp.Add or BinaryOp.Sub or BinaryOp.Mul or BinaryOp.Div or BinaryOp.Mod or BinaryOp.Pow,
        IsWildcard("lhs"),
        IsTensorConst("rhs", IsScalar()));

    private Expr? GetReplace(Binary binary, Expr lhs, TensorConst rhs)
    {
        return (binary.BinaryOp, rhs.Value.ToScalar<float>()) switch
        {
            (BinaryOp.Add, 0f) => lhs,
            (BinaryOp.Sub, 0f) => lhs,
            (BinaryOp.Mul, 1f) => lhs,
            (BinaryOp.Mod, 1f) => lhs,
            (BinaryOp.Pow, 1f) => lhs,
            (BinaryOp.Div, 1f) => lhs,
            _ => null,
        };
    }
}
