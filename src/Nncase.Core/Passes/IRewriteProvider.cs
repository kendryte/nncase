// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.Passes;

/// <summary>
/// Rewrite provider interface.
/// </summary>
public interface IRewriteProvider
{
    /// <summary>
    /// Rewrite expression.
    /// </summary>
    /// <param name="expr">Expression.</param>
    /// <param name="rules">Rewrite rules.</param>
    /// <param name="options">Options.</param>
    /// <returns>Rewrited expression.</returns>
    Expr Rewrite(Expr expr, IEnumerable<IRewriteRule> rules, RunPassContext options);
}

/// <summary>
/// EGraph Rewrite provider interface.
/// </summary>
public interface IEGraphRewriteProvider
{
    /// <summary>
    /// Using EGraph rewrite expression.
    /// </summary>
    /// <param name="expr">Expression.</param>
    /// <param name="rules">Rewrite rules.</param>
    /// <param name="options">Options.</param>
    /// <param name="compileOptions">compileOptions.</param>
    /// <returns>Rewrited expression.</returns>
    Expr ERewrite(Expr expr, IEnumerable<IRewriteRule> rules, RunPassContext options, CompileOptions compileOptions);

    /// <summary>
    /// Rewrite egraph.
    /// </summary>
    /// <param name="eGraph">EGraph.</param>
    /// <param name="rules">Rewrite rules.</param>
    /// <param name="options">Options.</param>
    /// <returns>Rewrited EGraph.</returns>
    IEGraph ERewrite(IEGraph eGraph, IEnumerable<IRewriteRule> rules, RunPassContext options);
}
