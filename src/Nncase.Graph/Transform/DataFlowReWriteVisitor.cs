// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Nncase.IR;
using Nncase.PatternMatch;

namespace Nncase.Transform
{
    /// <summary>
    /// DataFlowReWriteVisitor.
    /// </summary>
    internal sealed class DataFlowReWriteVisitor : ExprVisitor<Expr, IRType>
    {
        public ExprPattern Pattern { set; get; } = new InvalidPattern();

        public IRewriteRule Rule { set; get; } = new InvalidRule();

        /// <summary>
        /// a flag for fast exit, we can use it to know this rewrite results.
        /// </summary>
        public bool isMatched { private set; get; } = false;

        public override void Clear()
        {
            isMatched = false;
            base.Clear();
        }

        /// <summary>
        /// MatchCurExpr, in each node we try match pattern .
        /// </summary>
        /// <param name="expr"></param>
        /// <returns></returns>
        private Expr MatchCurExpr(Expr expr)
        {
            var matchs = DataFlowMatcher.Match(expr, Pattern);
            if (matchs.Count == 1)
            {
                isMatched = true;
                var res = Rule.GetReplace(matchs[0]);
                Pattern.Clear();
                return res ?? expr;
            }

            return expr;
        }

        /// <inheritdoc/>
        public override Expr DefaultVisitLeaf(Expr expr)
        {
            return MatchCurExpr(expr);
        }

        /// <inheritdoc/>
        public override Expr VisitLeaf(Call expr)
        {
            if (!isMatched)
            {
                return MatchCurExpr(expr);
            }

            return expr with
            {
                Target = ExpressionMemo[expr.Target],
                Parameters = (from p in expr.Parameters select ExpressionMemo[p]).ToImmutableArray(),
            };
        }

        /// <inheritdoc/>
        public override Expr VisitLeaf(Function expr)
        {
            if (!isMatched)
            {
                return MatchCurExpr(expr);
            }

            return expr with
            {
                Body = ExpressionMemo[expr.Body],
                Parameters = (from p in expr.Parameters select ExpressionMemo[p]).ToImmutableArray(),
            };
        }

        /// <inheritdoc/>
        public override Expr VisitLeaf(Tuple expr)
        {
            if (!isMatched)
            {
                return MatchCurExpr(expr);
            }

            return expr with
            {
                Fields = (from f in expr.Fields select ExpressionMemo[f]).ToImmutableArray(),
            };
        }
    }
}