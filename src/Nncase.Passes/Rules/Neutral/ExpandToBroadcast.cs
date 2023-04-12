// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Tensors;
using Nncase.PatternMatch;
using static Nncase.IR.F.Math;
using static Nncase.PatternMatch.Utility;
using Math = Nncase.IR.F.Math;

namespace Nncase.Passes.Rules.Neutral;

/// <summary>
/// Fold nop <see cref="IR.Math.Binary"/>.
/// </summary>
[RuleGenerator]
public sealed partial class ExpandToBroadcast : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } = PatternMatch.F.Tensors.IsExpand(
        "expand",
        "call",
        IsWildcard("input") with { TypePattern = TypePatternUtility.HasFixedShape() },
        IsTensorConst("shape"));

    private Expr? GetReplace(Expr input, TensorConst shape)
    {
        var inputRank = input.CheckedShape.Rank;
        var shapeSize = shape.Value.ToArray<int>().Length;
        var outputShape = Enumerable.Repeat((Expr)1, System.Math.Max(inputRank, shapeSize)).ToArray();
        for (var i = 0; i < shapeSize; i++)
        {
            outputShape[i + outputShape.Length - shapeSize] = shape.Value.ToArray<int>()[i];
        }

        for (int i = 0; i < inputRank; i++)
        {
            outputShape[i + outputShape.Length - inputRank] = Math.Max(input.CheckedShape[i].Value!, outputShape[i + outputShape.Length - inputRank]);
        }

        return IR.F.Tensors.Broadcast(input, IR.F.Tensors.Stack(new IR.Tuple(outputShape), 0));
    }
}
