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
            IsTensorConst("axis"));

    private Expr? GetReplace(Expr input, Call softmaxCall, int[] axis)
    {
        var exp = IR.F.Math.Exp(input);
        var reduce = input.CheckedDataType switch
        {
            var x when x == DataTypes.Float32 => IR.F.Tensors.ReduceSum(exp, axis, 0f, true),
            var x when x == DataTypes.Float64 => IR.F.Tensors.ReduceSum(exp, axis, 0d, true),
            var x when x == DataTypes.Float16 => IR.F.Tensors.ReduceSum(exp, axis, (Half)0f, true),
            var x when x == DataTypes.BFloat16 => IR.F.Tensors.ReduceSum(exp, axis, (BFloat16)0f, true),
            var x when x == DataTypes.Int32 => IR.F.Tensors.ReduceSum(exp, axis, 0, true),
            var x when x == DataTypes.Int64 => IR.F.Tensors.ReduceSum(exp, axis, 0L, true),
            var x when x == DataTypes.UInt32 => IR.F.Tensors.ReduceSum(exp, axis, 0u, true),
            var x when x == DataTypes.UInt64 => IR.F.Tensors.ReduceSum(exp, axis, 0ul, true),
            _ => throw new NotSupportedException(),
        };

        return IR.F.Math.Div(exp, reduce);
    }
}
