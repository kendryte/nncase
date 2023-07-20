// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using Nncase.IR;
using Nncase.PatternMatch;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.Utility;
using GetItem = Nncase.IR.Tensors.GetItem;

namespace Nncase.Passes.Rules.Neutral;

[RuleGenerator]
public partial class FoldNopReduce : RewriteRule<Pattern>
{
    public override Pattern Pattern => IsReduce(
        "reduceOp",
        "reduce",
        _ => true,
        IsWildcard("input") with { TypePattern = HasShape(new[] { 1 }) },
        IsTensorConst("axis"),
        IsTensorConst("initValue"),
        IsWildcard("keepDims"));

    private Expr? GetReplace(Expr input, Tensor keepDims)
    {
        if (keepDims.ToScalar<bool>())
        {
            return input;
        }

        return input[0];
    }
}
