// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using Nncase.Passes;
using Nncase.PatternMatch;
using static Nncase.IR.F.Math;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Neutral;

/// <summary>
/// squeeze to reshape.
/// </summary>
[RuleGenerator]
public sealed partial class FlattenToReshape : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } = IsFlatten(
        IsWildcard("input") with { TypePattern = HasFixedShape() },
        IsConstIntSclar("axis"));

    private Expr? GetReplace(Expr input, int axis)
    {
        var inShape = (RankedShape)input.CheckedShape;
        var postiveAxis = axis >= 0 ? axis : inShape.Rank + axis;
        var newShape = postiveAxis == 0 ? new RankedShape(1, inShape.Prod()) : new RankedShape(input.CheckedShape.ToValueArray()[..postiveAxis].Aggregate((a, b) => a * b), input.CheckedShape.ToValueArray()[postiveAxis..].Aggregate((a, b) => a * b));
        return Reshape(input, newShape);
    }
}
