// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.PatternMatch;
using System;
using System.Linq;
using Nncase.IR;
using Nncase.PatternMatch;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;

using GetItem = Nncase.IR.Tensors.GetItem;

namespace Nncase.Passes.Rules.ShapeExpr
{
    [RuleGenerator]
    public partial class FoldGetItemShapeOf : RewriteRule<Pattern>
    {
        public override Pattern Pattern => IsGetItem(null, "getItem", IsAlt(CastPattern, ShapeOfPattern), IsTensorConst("index"));

        public Pattern CastPattern => IsCast("cast", _ => true, ShapeOfPattern);

        public Pattern ShapeOfPattern => IsShapeOf(IsWildcard("input"));

        Expr? GetReplace(Expr input, Tensor<int> index, Call getItem)
        {
            DataType dt = DataTypes.Int64;

            if (getItem.Arguments[GetItem.Input.Index] is Call c && c.Target is IR.Tensors.Cast cast)
            {
                dt = cast.NewType;
            }

            if (index.Shape.IsScalar)
            {
                var dim = input.CheckedShape[index.ToScalar()];
                return dim.IsFixed ? IR.F.Tensors.Cast(dim.FixedValue, dt) : null;
            }

            return input;
        }
    }
}
