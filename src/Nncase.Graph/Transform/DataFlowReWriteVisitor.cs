using System.Linq;
using System.Collections.Generic;
using System.Collections.Immutable;
using Nncase.Pattern;
using Nncase.IR;

namespace Nncase.Transform
{

    internal sealed class DataFlowReWriteVisitor : ExprVisitor<Expr, IRType>
    {
        public ExprPattern Pattern { set; get; } = new InvalidPattern();

        public PatternRule Rule { set; get; } = new InvalidRule();

        public bool isMatched { private set; get; } = false;

        public override void Clear()
        {
            isMatched = false;
            base.Clear();
        }

        private Expr MatchCurExpr(Expr expr)
        {
            var matchs = DataFlowMatcher.Match(expr, Pattern);
            if (matchs.Count == 1)
            {
                isMatched = true;
                var res = Rule.GetRePlace(matchs[0]);
                Pattern.Clear();
                return res ?? expr;
            }
            return expr;
        }

        public override Expr DefaultVisitLeaf(Expr expr)
        {
            return MatchCurExpr(expr);
        }

        public override Expr VisitLeaf(Call expr)
        {
            if (!isMatched)
            {
                return MatchCurExpr(expr);
            }
            return expr with
            {
                Target = ExpressionMemo[expr.Target],
                Parameters = (from p in expr.Parameters select ExpressionMemo[p]).ToImmutableArray()
            };
        }

        public override Expr VisitLeaf(Function expr)
        {
            if (!isMatched)
            {
                return MatchCurExpr(expr);
            }
            return expr with
            {
                Body = ExpressionMemo[expr.Body],
                Parameters = (from p in expr.Parameters select ExpressionMemo[p]).ToImmutableArray()
            };
        }

        public override Expr VisitLeaf(Tuple expr)
        {
            if (!isMatched)
            {
                return MatchCurExpr(expr);
            }
            return expr with
            {
                Fields = (from f in expr.Fields select ExpressionMemo[f]).ToImmutableArray()
            };
        }
    }
}