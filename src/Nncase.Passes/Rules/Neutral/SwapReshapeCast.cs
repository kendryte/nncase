// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using Nncase.PatternMatch;
using static Nncase.IR.F.NN;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.TypePatternUtility;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.F.NN;
using static Nncase.PatternMatch.F.NN;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Neutral;

/// <summary>
/// Fold nop <see cref="IR.Tensors.Transpose"/>.
/// </summary>
[RuleGenerator]
public sealed partial class SwapReshapeCast : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } = IsCast(
            "cast",
            "castCall",
            _ => true,
            IsReshape(
                "reshape",
                "reshapeCall",
                _ => true,
                IsWildcard("input") with { TypePattern = IsFloat() },
                IsTensorConst("newShape")));

    private Expr? GetReplace(Expr input, Cast cast, Expr reshape, Expr newShape)
    {
        if (cast.NewType == DataTypes.Float16 && input.CheckedDataType == DataTypes.Float32)
        {
            return Reshape(Cast(input, DataTypes.Float16), newShape);
        }

        return null;
    }
}

/// <summary>
/// Fold nop <see cref="IR.Tensors.Transpose"/>.
/// </summary>
[RuleGenerator]
public sealed partial class SwapCastReshape : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } = IsReshape(
        "reshape",
        "reshapeCall",
        _ => true,
        IsCast(
            "cast",
            "castCall",
            _ => true,
            IsWildcard("input") with { TypePattern = IsFloat() }),
        IsTensorConst("newShape"));

    private Expr? GetReplace(Expr input, Cast cast, Expr newShape)
    {
        if (cast.NewType == DataTypes.Float32 && input.CheckedDataType == DataTypes.Float16)
        {
            return Cast(Reshape(input, newShape), DataTypes.Float32);
        }

        return null;
    }
}
