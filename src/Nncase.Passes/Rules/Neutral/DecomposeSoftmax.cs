// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.PatternMatch;
using Nncase.Utilities;

using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.F.NN;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Neutral;

/// <summary>
/// Decompose softmax.
/// </summary>
[RuleGenerator]
public sealed partial class DecomposeSoftmax : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } =
        IsSoftmax(
            "softmax",
            "softmaxCall",
            _ => true,
            IsWildcard("input"),
            IsFixedDimension("axis"));

    private Expr? GetReplace(Expr input, Call softmaxCall, int axis)
    {
        var normalizedaxes = new[] { axis < 0 ? axis + input.CheckedShape.Rank : axis };
        var max = input.CheckedDataType switch
        {
            var x when x == DataTypes.Float32 => IR.F.Tensors.ReduceMax(input, normalizedaxes, float.MinValue, true),
            var x when x == DataTypes.Float64 => IR.F.Tensors.ReduceMax(input, normalizedaxes, double.MinValue, true),
            var x when x == DataTypes.Float16 => IR.F.Tensors.ReduceMax(input, normalizedaxes, (Half)float.MinValue, true),
            var x when x == DataTypes.BFloat16 => IR.F.Tensors.ReduceMax(input, normalizedaxes, (BFloat16)float.MinValue, true),
            var x when x == DataTypes.Int32 => IR.F.Tensors.ReduceMax(input, normalizedaxes, int.MinValue, true),
            var x when x == DataTypes.Int64 => IR.F.Tensors.ReduceMax(input, normalizedaxes, long.MinValue, true),
            var x when x == DataTypes.UInt32 => IR.F.Tensors.ReduceMax(input, normalizedaxes, 0u, true),
            var x when x == DataTypes.UInt64 => IR.F.Tensors.ReduceMax(input, normalizedaxes, 0ul, true),
            _ => throw new NotSupportedException(),
        };

        var sub = input - max;
        var exp = IR.F.Math.Exp(sub);
        var reduce = input.CheckedDataType switch
        {
            var x when x == DataTypes.Float32 => IR.F.Tensors.ReduceSum(exp, normalizedaxes, 0f, true),
            var x when x == DataTypes.Float64 => IR.F.Tensors.ReduceSum(exp, normalizedaxes, 0d, true),
            var x when x == DataTypes.Float16 => IR.F.Tensors.ReduceSum(exp, normalizedaxes, (Half)0f, true),
            var x when x == DataTypes.BFloat16 => IR.F.Tensors.ReduceSum(exp, normalizedaxes, (BFloat16)0f, true),
            var x when x == DataTypes.Int32 => IR.F.Tensors.ReduceSum(exp, normalizedaxes, 0, true),
            var x when x == DataTypes.Int64 => IR.F.Tensors.ReduceSum(exp, normalizedaxes, 0L, true),
            var x when x == DataTypes.UInt32 => IR.F.Tensors.ReduceSum(exp, normalizedaxes, 0u, true),
            var x when x == DataTypes.UInt64 => IR.F.Tensors.ReduceSum(exp, normalizedaxes, 0ul, true),
            _ => throw new NotSupportedException(),
        };

        return IR.F.Math.Div(exp, reduce);
    }
}
