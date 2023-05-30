// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using DryIoc.ImTools;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using Nncase.PatternMatch;
using Nncase.Utilities;
using static Nncase.IR.F.Math;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Neutral;

/// <summary>
/// Fold two <see cref="IR.Tensors.Reshape"/>.
/// </summary>
[RuleGenerator]
public sealed partial class FoldTwoReduce : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } = IsReduce(
            "rd2",
            "rd2Call",
            _ => true,
            IsReduce(
                "rd1",
                "rd1Call",
                _ => true,
                IsWildcard("input"),
                IsTensorConst("axis1"),
                IsTensorConst("initValue1"),
                IsTensorConst("keepDims1")),
            IsTensorConst("axis2"),
            IsTensorConst("initValue2"),
            IsTensorConst("keepDims2")) with
    {
        TypePattern = HasFixedShape(),
    };

    private Expr? GetReplace(Expr input, Reduce rd1, Reduce rd2, int[] axis1, int[] axis2, float initValue1, float initValue2, bool keepDims1, bool keepDims2)
    {
        if (rd1.ReduceOp == rd2.ReduceOp
            && Math.Abs(initValue1 - initValue2) < float.Epsilon
            && keepDims1 == keepDims2
            && axis1.Length == 1
            && axis2.Length == 1
            && ((keepDims1 && (axis1[0] == -1 || axis1[0] == input.CheckedShape.Rank - 1) && (axis2[0] == -2 || axis2[0] == input.CheckedShape.Rank - 2))
            || ((axis1[0] == -1 || axis1[0] == input.CheckedShape.Rank - 1) && (axis2[0] == -1 || axis2[0] == input.CheckedShape.Rank - 2))))
        {
            return Reduce(rd1.ReduceOp, input, new[] { -2, -1 }, initValue1, keepDims1);
        }

        return null;
    }
}
