
using System;
using Nncase.IR;


namespace Nncase.Pattern
{

    public abstract class PatternRule
    {
        public virtual ExprPattern[] Patterns => new ExprPattern[] { Pattern };

        public ExprPattern Pattern { get; set; } = new InvalidPattern();

        public virtual Expr? GetRePlace(IMatchResult result) => throw new NotImplementedException("Not Implement GetRePlace!");
    }

    public sealed class InvalidRule : PatternRule
    {

    }
}