// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Linq;
using Nncase.IR;
using Nncase.PatternMatch;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.ShapeExpr;

[RuleGenerator]
public sealed partial class GatherToGetItem : RewriteRule<Pattern>
{
    // (Gather(input, 0, 0) -> GetItem(input)
    public override Pattern Pattern => IsGather("gather", 0, IsWildcard("input"), IsTensorConst("index") with { TypePattern = IsScalar() });

    private Expr? GetReplace(Expr input, int index)
    {
        return input[index];
    }
}
