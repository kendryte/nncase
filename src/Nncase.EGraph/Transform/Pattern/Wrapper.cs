using System;
using Nncase.IR;
namespace Nncase.Transform.Pattern
{
    public abstract record PatternWrapper()
    {
        private EMatchResult? _result { get; set; }

        protected T GetCast<T>(ExprPattern pattern) where T : Expr => ((T)(_result?[pattern] ?? throw new InvalidOperationException("Can't Get Expr When This Pattern Never Binding Result!")));

        public void Bind(EMatchResult result)
        {
            _result = result;
        }
    }

}