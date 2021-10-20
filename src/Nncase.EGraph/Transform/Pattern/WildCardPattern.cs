using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Nncase.IR;

namespace Nncase.Transform.Pattern
{
    public sealed record WildCardPattern(string Name, Func<Expr, bool> Cond) : ExprPattern
    {
        private static int _globalCardIndex = 0;

        public WildCardPattern() : this($"wc_{_globalCardIndex++}", x => (true))
        {
        }
        
        public WildCardPattern(string Name) : this(Name, x => (true))
        {
        }

        public override bool MatchLeaf(Expr expr) => Cond(expr) && MatchCheckedType(expr);
    }

    public static partial class Functional
    {
        public static WildCardPattern WildCard() => new WildCardPattern();
    }

}