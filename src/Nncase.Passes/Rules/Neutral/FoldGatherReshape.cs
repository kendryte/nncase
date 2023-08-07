// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Linq;
using Nncase.IR;
using Nncase.PatternMatch;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Neutral;

[RuleGenerator]
public sealed partial class FoldGatherReshape : RewriteRule<Pattern>
{
    // Reshape(Gather(Shape, 0, 0), new[] { 0 }) -> GetItem(Shape, 0)
    public override Pattern Pattern => IsGather(
        IsReshape(IsWildcard("input"), IsTensorConst("newShape")), IsTensorConst("axis"), IsTensorConst("index"));

    private Expr? GetReplace(Expr input, int[] newShape, int axis, int index)
    {
        if (newShape.SequenceEqual(new[] { 1 }) && axis == 1)
        {
            return input[index];
        }

        return null;
    }
}
