using System.Collections.Generic;
using System;
using System.Collections.Immutable;
using System.Linq;
using Nncase.IR;

namespace Nncase.Pattern
{
    /// <summary>
    /// Wildcard Pattern Can Match Any Thing
    /// </summary>
    /// <param name="Name"></param>
    public sealed record WildCardPattern(string Name) : ExprPattern
    {

        /// <summary>
        /// Initializes a new instance of the <see cref="WildCardPattern"/> class.
        /// </summary>
        public WildCardPattern() : this($"wc") { }

        /// <summary>
        /// Cast the Name to Pattern
        /// </summary>
        /// <param name="Name"></param>
        public static implicit operator WildCardPattern(string Name) => new WildCardPattern(Name);

        /// <inheritdoc/>
        public override bool MatchLeaf(Expr expr) => MatchCheckedType(expr);
    }

    /// <summary>
    /// the invalid pattern
    /// </summary>
    public sealed record InvalidPattern : ExprPattern
    {

    }

    public static partial class Utility
    {

        /// <summary>
        /// fast utility for build wildcard pattern
        /// </summary>
        /// <returns> Returns. </returns>
        public static WildCardPattern IsWildCard() => new WildCardPattern();

        /// <summary>
        /// 
        /// </summary>
        /// <param name="Name"></param>
        /// <returns> Returns. </returns>
        public static WildCardPattern IsWildCard(string Name) => new WildCardPattern(Name);
    }

}