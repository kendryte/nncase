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

/// <summary>
/// Fold <see cref="IR.Math.Clamp"/> by const range.
/// </summary>
[RuleGenerator]
public sealed partial class FoldClampByRangeConst : IRewriteRule
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

/// <summary>
/// Fold <see cref="IR.Math.Clamp"/> by range var.
/// </summary>
[RuleGenerator]
public sealed partial class FoldClampByRangeVar : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } = IsClamp(
        IsWildcard("input"),
        IsTensorConst("min"),
        IsWildcard("max"));

    private Expr? GetReplace(Expr input, Tensor min, Expr max)
    {
        var maxValue = max.Metadata.Range?.Max;
        if (maxValue is null)
        {
            return null;
        }

        if (input.Metadata.Range!.Value.Max < maxValue && min.ToArray<float>()[0] <= input.Metadata.Range!.Value.Min)
        {
            return input;
        }

        return null;
    }
}