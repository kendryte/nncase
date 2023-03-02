// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using Nncase.IR;

namespace Nncase.Passes;

/// <summary>
/// The Usedby.
/// </summary>
public interface IUsedByResult
{
    /// <summary>
    /// Gets get the memo.
    /// </summary>
    IReadOnlyDictionary<Expr, HashSet<Expr>> MeMo { get; }

    /// <summary>
    /// get the which parent used child expr node.
    /// </summary>
    HashSet<Expr> Get(Expr child);

    /// <summary>
    /// clear usedby information.
    /// </summary>
    /// <param name="child">child expressions.</param>
    /// <param name="parent">parent expressions.</param>
    void Clear(Expr child, Expr parent);

    /// <summary>
    /// get usedby information.
    /// </summary>
    /// <param name="child">child expressions.</param>
    /// <param name="parent">parent expressions.</param>
    void Add(Expr child, Expr parent);

    /// <summary>
    /// transfer the usedy information.
    /// </summary>
    void Transfer(Expr old_expr, Expr new_expr);
}
