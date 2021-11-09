
using System;
using Nncase.IR;
using Nncase.Transform.Pattern;

namespace Nncase.Transform.Rule
{

    public abstract class EGraphRule
    {
        public virtual ExprPattern[] GetPatterns()
        {
            return new ExprPattern[] { Pattern };
        }

        public ExprPattern Pattern { get; set; }


        public virtual Expr? GetRePlace(EMatchResult result) => throw new NotImplementedException("Not Implement GetRePlace!");
    }
}