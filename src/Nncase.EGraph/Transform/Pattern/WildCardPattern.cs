using System.Collections.Generic;
using System;
using System.Collections.Immutable;
using System.Linq;
using Nncase.IR;

namespace Nncase.Transform.Pattern
{
    public sealed record WildCardPattern(ID Id) : ExprPattern(Id)
    {
        public WildCardPattern() : this(Utility.GetID())
        {
        }

        public static implicit operator WildCardPattern(string Prefix) => Utility.IsWildCard(Prefix);

        public override bool MatchLeaf(Expr expr) => MatchCheckedType(expr);

    }

    public static partial class Utility
    {
        public static WildCardPattern IsWildCard(ID Id) => new WildCardPattern(Id);

        public static WildCardPattern IsWildCard(string Prefix) => new WildCardPattern(GetID(Prefix));

        public static WildCardPattern IsWildCard() => IsWildCard(GetID());
    }

}