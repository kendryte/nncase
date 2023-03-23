// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using Nncase.PatternMatch;
using static Nncase.IR.F.Math;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Neutral;

/// <summary>
/// Fold nop <see cref="IR.Math.Clamp"/>.
/// </summary>
[RuleGenerator]
public sealed partial class FoldNopClamp : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } = IsClamp(
        IsWildcard("input"),
        IsTensorConst("min", HasDataType(DataTypes.Float32)),
        IsTensorConst("max", HasDataType(DataTypes.Float32)));

    private Expr? GetReplace(Expr input, Tensor<float> min, Tensor<float> max)
    {
        if (min.All(v => v <= float.MinValue) && max.All(v => v >= float.MaxValue))
        {
            return input;
        }

        return null;
    }
}
