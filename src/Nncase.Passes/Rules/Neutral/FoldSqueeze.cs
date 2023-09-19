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

[RuleGenerator]
public partial class FoldUnsqueezeSqueeze : RewriteRule<Pattern>
{
    /// <inheritdoc/>
    public override Pattern Pattern => IsUnsqueeze(
        "unsqu",
        "output",
        IsSqueeze(IsWildcard("input"), IsTensorConst("sqAxes")),
        IsTensorConst("unsqAxes"));

    private Expr? GetReplace(Call output, Expr input)
    {
        if (output.CheckedShape.SequenceEqual(input.CheckedShape))
        {
            return input;
        }

        return null;
    }
}

[RuleGenerator]
public partial class FoldSqueezeUnsqueeze : RewriteRule<Pattern>
{
    /// <inheritdoc/>
    public override Pattern Pattern => IsSqueeze(
        "sqOp",
        "output",
        IsUnsqueeze(IsWildcard("input"), IsTensorConst("unsqAxes")),
        IsTensorConst("sqAxes"));

    private Expr? GetReplace(Call output, Expr input)
    {
        if (output.CheckedShape.SequenceEqual(input.CheckedShape))
        {
            return input;
        }

        return null;
    }
}
