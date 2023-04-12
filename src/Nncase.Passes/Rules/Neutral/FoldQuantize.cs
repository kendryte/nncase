// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.PatternMatch;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Neutral;

/// <summary>
/// Fold  input -> quant -> dequant.
/// </summary>
[RuleGenerator]
public sealed partial class FoldQuantDeQuant : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } = IsDequantize(
        "dequantize",
        "output",
        x => true,
        IsQuantize(
            "q",
            string.Empty,
            x => true,
            IsWildcard("input"),
            IsTensorConst("q_param")),
        IsTensorConst("deq_param"));

    /// <summary>
    /// the quant param almost equal.
    /// </summary>
    public static bool AlmostEqual(QuantParam lhs, QuantParam rhs)
    {
        return lhs.ZeroPoint == rhs.ZeroPoint && System.MathF.Abs(lhs.Scale - rhs.Scale) <= float.Epsilon;
    }

    private Expr? GetReplace(Expr input, QuantParam q_param, QuantParam deq_param, IR.Math.Dequantize dequantize)
    {
        if (AlmostEqual(q_param, deq_param) && input.CheckedDataType == dequantize.TargetType)
        {
            return input;
        }

        return null;
    }
}

/// <summary>
/// Fold  input -> dequant -> quant.
/// </summary>
[RuleGenerator]
public sealed partial class FoldDeQuantQuant : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } = IsQuantize(
        "quantize",
        "output",
        x => true,
        IsDequantize(
            "deq",
            string.Empty,
            x => true,
            IsWildcard("input"),
            IsTensorConst("deq_param")),
        IsTensorConst("q_param"));

    private Expr? GetReplace(Expr input, QuantParam q_param, QuantParam deq_param, IR.Math.Quantize quantize)
    {
        if (FoldQuantDeQuant.AlmostEqual(q_param, deq_param) && input.CheckedDataType == quantize.TargetType)
        {
            return input;
        }

        return null;
    }
}
