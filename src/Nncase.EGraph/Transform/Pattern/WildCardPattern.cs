using System.Collections.Generic;
using System;
using System.Collections.Immutable;
using System.Linq;
using Nncase.IR;

namespace Nncase.Transform.Pattern
{
    public sealed record WildCardPattern(string Name) : ExprPattern
    {

        public WildCardPattern() : this($"wc") { }

        public static implicit operator WildCardPattern(string Name) => new WildCardPattern(Name);

        public override bool MatchLeaf(Expr expr) => MatchCheckedType(expr);
    }

    public static partial class Utility
    {
        public static WildCardPattern IsWildCard() => new WildCardPattern();

        public static WildCardPattern IsWildCard(string Name) => new WildCardPattern(Name);
    }

}