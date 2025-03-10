// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.NN;
using Nncase.Passes.Transforms;
using Nncase.PatternMatch;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Neutral;

[RuleGenerator]
public sealed partial class InferRange : RewriteRule<Pattern>
{
    public override Pattern Pattern => IsWildcard("expr", expr => expr.Metadata.Range is null);

    private Expr GetReplace(Expr expr)
    {
        var visitor = new InferRangeVisitor();
        visitor.Visit(expr);
        return expr;
    }
}
