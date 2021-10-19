using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Nncase.IR;

namespace Nncase.Transform.Pattern
{
    public sealed record WildCardPattern(Func<Expr, bool> Cond) : ExprPattern
    {
        public bool MatchLeaf(Expr expr) => Cond(expr) && MatchCheckedType(expr);
    }

    public static partial class Functional
    {
        public static WildCardPattern WildCard => new WildCardPattern(x => (true));
    }

}