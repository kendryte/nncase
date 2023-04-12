// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Nncase.IR;
using Nncase.PatternMatch;
using Tensorflow;
using static Nncase.IR.F.NN;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.Utility;
using Shape = Nncase.IR.Shape;

namespace Nncase.Passes.Rules.Neutral;

/// <summary>
/// Transform <see cref="IR.Math.MatMul"/> to <see cref="IR.NN.Conv2D"/>.
/// </summary>
[RuleGenerator]
public sealed partial class ReshapeBatchMatmul : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } = IsMatMul(
        "mm",
        "call",
        _ => true,
        IsWildcard("a") with { TypePattern = (HasRank(3) | HasRank(4)) & HasFixedShape() },
        IsWildcard("b") with { TypePattern = HasFixedShape() });

    private Expr? GetReplace(Call call, Expr a, Expr b)
    {
        var aShape = a.CheckedShape;
        var bShape = b.CheckedShape;
        if (aShape[^2] != 1 || bShape.Size != bShape[^2] * bShape[^1])
        {
            return null;
        }

        var newAShape = new Shape(aShape.Size / aShape[^1], aShape[^1]);
        var newBShape = new Shape(bShape[^2], bShape[^1]);

        return Reshape(
            MatMul(
                Reshape(a, newAShape),
                Reshape(b, newBShape)),
            call.CheckedShape);
    }
}
