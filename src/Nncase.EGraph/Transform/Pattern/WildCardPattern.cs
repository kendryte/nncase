using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Nncase.IR;

namespace Nncase.Transform.Pattern
{
    public sealed record WildCardPattern(string Name, ExprPattern? Pattern) : ExprPattern
    {
        private static int _globalCardIndex = 0;

        public WildCardPattern() : this($"wc_{_globalCardIndex++}", null)
        {
        }

        public static implicit operator WildCardPattern(string Name) => new WildCardPattern(Name, null);

        public override bool MatchLeaf(Expr expr) => (Pattern?.MatchLeaf(expr) ?? true) && MatchCheckedType(expr);

        public WildCardPattern Dup(int index) => new WildCardPattern($"{this.Name}_{index}", this.Pattern)
        {
            CheckedTypePat = this.CheckedTypePat
        };
    }

    public static partial class Utility
    {
        public static WildCardPattern IsWildCard() => new WildCardPattern();

        public static VArgsPattern IsVArgsWildCard(string Name, ExprPattern? pattern)
          => new VArgsPattern(new WildCardPattern(Name, pattern)) { Repeat = true };
    }

}