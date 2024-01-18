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
        IsTensorConst("rhs"));

    private Expr? GetReplace(Binary binary, Expr lhs, TensorConst rhs)
    {
        return binary.BinaryOp switch
        {
            BinaryOp.Add when rhs.Value.ToArray<float>().All(x => x == 0) => lhs,
            BinaryOp.Sub when rhs.Value.ToArray<float>().All(x => x == 0) => lhs,
            BinaryOp.Mul when rhs.Value.ToArray<float>().All(x => x == 1) => lhs,
            BinaryOp.Pow when rhs.Value.ToArray<float>().All(x => x == 1) => lhs,
            BinaryOp.Div when rhs.Value.ToArray<float>().All(x => x == 1) => lhs,
            _ => null,
        };
    }
}
