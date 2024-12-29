// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reactive;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.NN;
using Nncase.PatternMatch;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Neutral;

[RuleGenerator]
public sealed partial class LiftCEInIf : RewriteRule<Pattern>
{
    public override Pattern Pattern => IsWildcard("expr", expr => expr is If);

    private Expr? GetReplace(If expr)
    {
        var parameters = expr.ParamList.ToList();
        var thenExprs = LiftCollector.Collect(expr.Then).ToHashSet(ReferenceEqualityComparer.Instance);
        var elseExprs = LiftCollector.Collect(expr.Else).ToHashSet(ReferenceEqualityComparer.Instance);
        var commonExprs = thenExprs.Intersect(elseExprs).Where(x => !(x is Var or If)).Except(parameters).Cast<Expr>().ToArray();
        if (commonExprs.Any())
        {
            parameters.AddRange(commonExprs);
            return new If(expr.Condition, expr.Then, expr.Else, parameters.ToArray());
        }
        else
        {
            return null;
        }
    }

    public sealed class LiftCollector : ExprWalker<List<Expr>>
    {
        private LiftCollector()
        {
        }

        public static IReadOnlyList<Expr> Collect(Expr expr)
        {
            var exprs = new List<Expr>();
            new LiftCollector().Visit(expr, exprs);
            return exprs;
        }

        // protected override Unit VisitIf(If expr, List<Expr> context)
        // {
        //     foreach (var param in expr.ParamList)
        //     {
        //         Visit(param, context);
        //     }
        //     Visit(expr.Condition, context);
        //     return default;
        // }
        protected override Unit DefaultVisitLeaf(Expr expr, List<Expr> context)
        {
            context.Add(expr);
            return default;
        }
    }
}
