// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Toolkit.HighPerformance.Helpers;

namespace Nncase.IR;

/// <summary>
/// Leaf expression equality comparer.
/// </summary>
public sealed class LeafExprEqualityComparer : IEqualityComparer<Expr>
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
            (Fusion tx, Fusion ty) => tx.Equals(ty),

            // note think of primfunc/primfunc wrapper as a black box.
            (TIR.PrimFunction tx, TIR.PrimFunction ty) => ReferenceEquals(tx, ty),
            (PrimFunctionWrapper tx, PrimFunctionWrapper ty) => ReferenceEquals(tx, ty),
            (Function tx, Function ty) => tx.Parameters.Length == ty.Parameters.Length,
            (Tuple tx, Tuple ty) => tx.Count == ty.Count,
            (Call tx, Call ty) => tx.Arguments.Length == ty.Arguments.Length,
            (Op tx, Op ty) => tx.Equals(ty),
            (IR.If, IR.If) => true,
            (Marker tx, Marker ty) => tx.Name == ty.Name,
            (None tx, None ty) => tx.Equals(ty),
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
            Function x => ReferenceEqualityComparer.Instance.GetHashCode(x),
            Fusion x => x.GetHashCode(),
            TIR.PrimFunction x => ReferenceEqualityComparer.Instance.GetHashCode(x),
            PrimFunctionWrapper x => ReferenceEqualityComparer.Instance.GetHashCode(x),
            Tuple x => x.Count.GetHashCode(),
            Call x => x.Arguments.Length.GetHashCode(),
            Op x => x.GetHashCode(),
            Marker x => x.Name.GetHashCode(StringComparison.Ordinal),
            None x => x.GetHashCode(),
            IR.If x => x.GetType().GetHashCode(),
            _ => throw new InvalidOperationException("Invalid expression type."),
        };
    }
}
