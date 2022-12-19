using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using Nncase.IR;

namespace Nncase.Transform;

/// <summary>
/// The Usedby
/// </summary>
public interface IUsedByResult
{
    /// <summary>
    /// get the which parent used child expr node.
    /// </summary>
    /// <param name="child"></param>
    /// <returns></returns>
    HashSet<Expr> Get(Expr child);

    /// <summary>
    /// clear usedby information
    /// </summary>
    /// <param name="child">child expressions.</param>
    /// <param name="parent">parent expressions.</param>
    void Clear(Expr child, Expr parent);

    /// <summary>
    /// get usedby information
    /// </summary>
    /// <param name="child">child expressions.</param>
    /// <param name="parent">parent expressions.</param>
    void Add(Expr child, Expr parent);

    /// <summary>
    /// transfer the usedy information.
    /// </summary>
    /// <param name="old_expr"></param>
    /// <param name="new_expr"></param>
    void Transfer(Expr old_expr, Expr new_expr);

    /// <summary>
    /// Get the memo.
    /// </summary>
    IReadOnlyDictionary<Expr, HashSet<Expr>> MeMo { get; }
}
