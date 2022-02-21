// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.Evaluator;
using Nncase.IR;
using Nncase.IR.Tensors;
using Nncase.PatternMatch;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.F.NN;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Transform.Rule
{
    public class FoldConstCall : IRewriteRule
    {
        public FoldConstCall()
        {
            Pattern = IsCall(IsWildcard(), IsVArgsRepeat(() => IsAlt(IsConst(), IsConstTuple())));
        }

        public override Expr? GetReplace(IMatchResult result)
        {
            var expr = result[Pattern];
            return Const.FromValue(expr.Evaluate());
        }
    }

    public class FoldConstFunction : IRewriteRule
    {
        public FoldConstFunction()
        {
            Pattern = IsFunction(IsWildcard(), IsVArgsRepeat(() => IsAlt(IsConst(), IsConstTuple())));
        }

        public override Expr? GetReplace(IMatchResult result) => Const.FromValue(result[Pattern].Evaluate());
    }

    public class FoldShapeOp : IRewriteRule
    {
        WildcardPattern wc = "input";

        public FoldShapeOp()
        {
            Pattern = ShapeOp(wc);
        }

        public override Expr? GetReplace(IMatchResult result)
        {
            return Const.FromShape(result[wc].CheckedShape);
        }
    }
}
