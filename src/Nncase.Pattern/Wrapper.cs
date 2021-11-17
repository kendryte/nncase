using System;
using Nncase.IR;

namespace Nncase.Pattern
{
    public abstract record PatternWrapper()
    {
        private IMatchResult? _result { get; set; }

        protected T GetCast<T>(ExprPattern pattern) where T : Expr => ((T)(_result?[pattern] ?? throw new InvalidOperationException("Can't Get Expr When This Pattern Never Binding Result!")));

        public void Bind(IMatchResult result)
        {
            _result = result;
        }
    }

}