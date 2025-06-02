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
        "call",
        x => x.BinaryOp is BinaryOp.Add or BinaryOp.Sub or BinaryOp.Mul or BinaryOp.Div or BinaryOp.Mod or BinaryOp.Pow,
        IsWildcard("lhs"),
        IsTensorConst("rhs"));

    private Expr? GetReplace(Binary binary, Call call, Expr lhs, TensorConst rhs)
    {
        if ((lhs.CheckedType is Nncase.IR.AnyType && rhs.CheckedShape.IsScalar) || (lhs.CheckedShape == call.CheckedShape))
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

        return null;
    }
}

/// <summary>
/// Fold nop <see cref="IR.Math.Binary"/>.
/// </summary>
[RuleGenerator]
public sealed partial class FoldSameBinary : IRewriteRule
{
    private readonly Pattern _operandPattern = IsWildcard("operand");

    public FoldSameBinary()
    {
        Pattern = IsBinary(
            "binary",
            "call",
            x => x.BinaryOp is BinaryOp.Min or BinaryOp.Max,
            _operandPattern,
            _operandPattern);
    }

    /// <inheritdoc/>
    public IPattern Pattern { get; }

    private Expr? GetReplace(Expr operand)
    {
        return operand;
    }
}

/// <summary>
/// Fold nop <see cref="IR.Math.Binary"/> by range.
/// </summary>
[RuleGenerator]
public sealed partial class FoldNopBinaryByRange : IRewriteRule
{
    public IPattern Pattern { get; } = IsBinary(
        "binary",
        "call",
        x => x.BinaryOp is BinaryOp.Min or BinaryOp.Max,
        IsWildcard("lhs"),
        IsTensorConst("rhs"));

    private Expr? GetReplace(Binary binary, Expr lhs, TensorConst rhs)
    {
        if (lhs.Metadata.Range is null)
        {
            return null;
        }

        var lhsRangeMin = lhs.Metadata.Range.Value.Min;
        var lhsRangeMax = lhs.Metadata.Range.Value.Max;

        if (rhs.CheckedShape.IsScalar)
        {
            return binary.BinaryOp switch
            {
                BinaryOp.Min when rhs.Value.ToArray<float>().All(x => x >= lhsRangeMax) => lhs,
                BinaryOp.Max when rhs.Value.ToArray<float>().All(x => x <= lhsRangeMin) => lhs,
                _ => null,
            };
        }

        return null;
    }
}
