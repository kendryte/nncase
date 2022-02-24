// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.IR;

/// <summary>
/// Leaf expression equality comparer.
/// </summary>
public class LeafExprEqualityComparer : IEqualityComparer<Expr>
{
    /// <summary>
    /// Gets instance.
    /// </summary>
    public static LeafExprEqualityComparer Instance { get; } = new();

    /// <inheritdoc/>
    public bool Equals(Expr? x, Expr? y)
    {
        if (x == null && y == null)
        {
            return true;
        }

        if (x == null || y == null)
        {
            return false;
        }

        if (x.GetType() != y.GetType())
        {
            return false;
        }

        return (x, y) switch
        {
            (Var tx, Var ty) => tx.Equals(ty),
            (Const tx, Const ty) => tx.Equals(ty),
            (Function tx, Function ty) => tx.Parameters.Equals(ty.Parameters),
            (Tuple tx, Tuple ty) => tx.Count == ty.Count,
            (Call tx, Call ty) => tx.Parameters.Count == ty.Parameters.Count,
            (Op tx, Op ty) => tx.Equals(ty),
            _ => throw new InvalidOperationException("Invalid expression type."),
        };
    }

    /// <inheritdoc/>
    public int GetHashCode(Expr obj)
    {
        return obj switch
        {
            Var x => x.GetHashCode(),
            Const x => x.GetHashCode(),
            Function x => x.Parameters.GetHashCode(),
            Tuple x => x.Count.GetHashCode(),
            Call x => x.Parameters.Count.GetHashCode(),
            Op x => x.GetHashCode(),
            _ => throw new InvalidOperationException("Invalid expression type."),
        };
    }
}
