// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.TIR;

/// <summary>
/// A for loop, with poissible type annotations.
/// <example>
/// <code>
///   for (loop_var = min; loop_var &lt; min + extent; ++loop_var) {
///     body
///    }
/// </code>
/// </example>
/// </summary>
/// <param name="LoopVar">The loop variable.</param>
/// <param name="Domain">The domain of for range.</param>
/// <param name="Mode">The kind of the for loop.</param>
/// <param name="Body">the body sequence.</param>
public sealed record For(Var LoopVar, Range Domain, LoopMode Mode, Sequential Body) : Expr
{
    /// <summary>
    /// Initializes a new instance of the <see cref="For"/> class.
    /// </summary>
    /// <param name="loopVar">The loop variable.</param>
    /// <param name="domain">The domain of for range.</param>
    /// <param name="mode">The kind of the for loop.</param>
    public For(Var loopVar, Range domain, LoopMode mode)
        : this(loopVar, domain, mode, new())
    {
    }
}
