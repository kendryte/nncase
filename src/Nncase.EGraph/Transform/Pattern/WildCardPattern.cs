using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Nncase.IR;

namespace Nncase.Transform.Pattern
{
    public sealed record WildCardPattern(string Name, ExprPattern? SubPattern) : ExprPattern
    {
        private static int _globalCardIndex = 0;

        public WildCardPattern() : this($"wc_{_globalCardIndex++}", null)
        {
        }

        public static implicit operator WildCardPattern(string Name) => new WildCardPattern(Name, null);

        public override bool MatchLeaf(Expr expr) => MatchCheckedType(expr);

        public override WildCardPattern Dup(string Suffix)
          => new WildCardPattern(
            $"{this.Name}_{Suffix}",
            this.SubPattern?.Dup($"{Suffix}"))
          {
              CheckedTypePat = this.CheckedTypePat
          };
    }

    public static partial class Utility
    {
        public static WildCardPattern IsWildCard() => new WildCardPattern();
        public static WildCardPattern IsWildCard(string Name) => new WildCardPattern(Name, null);
        public static WildCardPattern IsWildCard(string Name, ExprPattern SubPattern) => new WildCardPattern(Name, SubPattern);

        // public static VArgsPattern IsVArgsWildCard(string Name, ExprPattern? pattern)
        //   => new VArgsPattern(new WildCardPattern(Name, pattern)) { Repeat = true };
    }

}