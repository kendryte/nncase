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
    public override Pattern Pattern => IsGetItem(null, "getItem", IsAlt(CastPattern, ShapeOfPattern), IsAlt("index", IsFixedDimension(), IsFixedShape()));

    public Pattern CastPattern => IsCast("cast", _ => true, ShapeOfPattern);

    public Pattern ShapeOfPattern => IsShapeOf(IsWildcard("input") with { TypePattern = HasRankedShape() });

    private Expr? GetReplace(Expr input, Tensor<int> index, Call getItem)
    {
        DataType dt = DataTypes.Int64;

        if (getItem[GetItem.Input] is Call c && c.Target is IR.Tensors.Cast cast)
        {
            dt = cast.NewType;
        }

        if (index.Length == 1)
        {
            var dim = IR.F.Shapes.AsTensor(input.CheckedShape[index.Single()]);
            return dt == DataTypes.Int64 ? dim : IR.F.Tensors.Cast(dim, dt);
        }

        return null;
    }
}
