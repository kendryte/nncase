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
using static Nncase.Passes.Rules.Neutral.FoldSqueezeCommon;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Neutral;

[RuleGenerator]
public partial class FoldUnsqueezeSqueeze : RewriteRule<Pattern>
{
    /// <inheritdoc/>
    public override Pattern Pattern => IsUnsqueeze(
        IsSqueeze(IsWildcard("input"), IsTensorConst("sqAxes")),
        IsTensorConst("unsqAxes"));

    private Expr? GetReplace(Expr input, int[] sqAxes, int[] unsqAxes)
    {
        var r = input.CheckedShape.Rank;
        if (!CanFold(sqAxes, unsqAxes, r))
        {
            return null;
        }

        return input;
    }
}

[RuleGenerator]
public partial class FoldSqueezeUnsqueeze : RewriteRule<Pattern>
{
    /// <inheritdoc/>
    public override Pattern Pattern => IsSqueeze(
        IsUnsqueeze(IsWildcard("input"), IsTensorConst("unsqAxes")),
        IsTensorConst("sqAxes"));

    // todo: maybe error
    private Expr? GetReplace(Expr input, int[] sqAxes, int[] unsqAxes)
    {
        var r = input.CheckedShape.Rank;
        if (!CanFold(sqAxes, unsqAxes, r))
        {
            return null;
        }

        return input;
    }
}

// used for dynamic shape
internal static class FoldSqueezeCommon
{
    internal static bool CanFold(int[] sqAxes, int[] unsqAxes, int rank)
    {
        // now only support same axes for dynamic shape
        if (sqAxes.Length != unsqAxes.Length)
        {
            return false;
        }

        // positive
        var positiveSqAxes = sqAxes.Select(x => x < 0 ? x + rank : x).ToArray();
        var positiveUnsqAxes = unsqAxes.Select(x => x < 0 ? x + rank : x).ToArray();
        return positiveSqAxes.SequenceEqual(positiveUnsqAxes);
    }
}
