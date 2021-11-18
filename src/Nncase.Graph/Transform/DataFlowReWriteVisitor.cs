using System.Collections.Generic;
using Nncase.IR;
using Nncase.Pattern;

namespace Nncase.Transform
{

    internal sealed class DataFlowReWriteVisitor : ExprVisitor<Expr, IRType>
    {
        public ExprPattern Pattern { set; get; } = new InvalidPattern();

        public PatternRule Rule { set; get; } = new InvalidRule();

        public bool isMatched { private set; get; } = false;

        public void Clear()
        {
            isMatched = false;
        }

        public override Expr DefaultVisitLeaf(Expr expr)
        {
            if (!isMatched)
            {
                var matchs = DataFlowMatcher.Match(expr, Pattern);
                if (matchs.Count == 1)
                {
                    isMatched = true;
                    return Rule.GetRePlace(matchs[0]) ?? expr;
                }
            }
            return expr;
        }

    }
}