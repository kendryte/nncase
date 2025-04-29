// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using Nncase.IR;
using Nncase.PatternMatch;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;
using GetItem = Nncase.IR.Tensors.GetItem;

namespace Nncase.Passes.Rules.ShapeExpr;

[RuleGenerator]
public partial class FoldGetItemShapeOf : RewriteRule<Pattern>
{
    public override Pattern Pattern => IsGetItem(null, "getItem", ShapeOfPattern, IsTensorConst("index", IsScalar() | HasShape(new RankedShape(1L))));

    public Pattern ShapeOfPattern => IsShapeOf(IsWildcard("input") with { TypePattern = HasRankedShape() });

    private Expr? GetReplace(Expr input, Tensor<int> index, Call getItem)
    {
        return IR.F.Shapes.AsTensor(input.CheckedShape[index.Single()]);
    }
}
