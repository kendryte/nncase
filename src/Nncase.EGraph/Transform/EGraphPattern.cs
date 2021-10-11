// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.Transform.EGraphPatterns
{
    /// <summary>
    /// EGraph pattern.
    /// </summary>
    public abstract record EGraphPattern
    {
    }

    /// <summary>
    /// Wildcard pattern.
    /// </summary>
    public sealed record WildcardPattern() : EGraphPattern;

    /// <summary>
    /// Variable pattern.
    /// </summary>
    public sealed record VarPattern() : EGraphPattern
    {
    }

    public sealed record ConstPattern() : EGraphPattern
    {
    }

    /// <summary>
    /// Functional patterns.
    /// </summary>
    public static class Functional
    {
        /// <summary>
        /// Wildcard.
        /// </summary>
        public static readonly WildcardPattern Wildcard = new WildcardPattern();
    }
}
