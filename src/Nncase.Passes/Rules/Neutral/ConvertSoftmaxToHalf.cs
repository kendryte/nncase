// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.IR;
using Nncase.PatternMatch;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.F.NN;
using static Nncase.PatternMatch.Utility;

using static Nncase.IR.F.NN;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.F.NN;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Neutral;

/// <summary>
/// Fold nop <see cref="IR.Tensors.Transpose"/>.
/// </summary>
[RuleGenerator]
public sealed partial class ConvertSoftmaxToHalf : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } = IsSoftmax(
            "softmax",
            "softmaxCall",
            _ => true,
            IsWildcard("input") with { TypePattern = IsFloat() },
            IsTensorConst("axis"));

    private Expr? GetReplace(Expr input, Expr axis)
    {
        if (input.CheckedDataType != DataTypes.Float16)
        {
            var inputHalf = Cast(input, DataTypes.Float16);
            var newSoftmax = Softmax(inputHalf, axis);
            var output = Cast(newSoftmax, input.CheckedDataType);

            return output;
        }

        return null;
    }
}
