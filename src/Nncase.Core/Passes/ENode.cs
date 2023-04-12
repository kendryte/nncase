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
/// ENode.
/// </summary>
public sealed record ENode
{
    /// <summary>
    /// speedup hashcode calc.
    /// </summary>
    private int? _hashcode;

    private ENode(Expr expr, IRArray<EClass> children)
    {
        Expr = expr;
        Children = children;
    }

    /// <summary>
    /// Gets the Enode's Expression.
    /// </summary>
    public Expr Expr { get; init; }

    /// <summary>
    /// Gets the Enode Children Eclasses.
    /// </summary>
    public IRArray<EClass> Children { get; init; }

    /// <summary>
    /// The create for the enode.
    /// Can add hook in here for debug.
    /// </summary>
    /// <param name="expr">expression.</param>
    /// <param name="children">parameters.</param>
    public static ENode Create(Expr expr, IRArray<EClass> children) => new ENode(expr, children);

    /// <summary>
    /// Add current enode information to childrens.
    /// </summary>
    /// <param name="eClass">EClass.</param>
    public void SetClass(EClass eClass)
    {
        eClass.AddNode(this);

        foreach (var child in Children)
        {
            child.AddUsedBy(this);
        }
    }

    /// <summary>
    /// Canonicalize this enode.
    /// </summary>
    /// <returns>Canonicalized enode.</returns>
    public ENode Canonicalize()
    {
        var children = (from c in Children select c.Find()).ToArray();
        return ENode.Create(Expr, children);
    }

    /// <inheritdoc/>
    public bool Equals(ENode? other)
    {
        return !(other is null)
            && LeafExprEqualityComparer.Instance.Equals(Expr, other.Expr)
            && Children.Equals(other.Children);
    }

    /// <inheritdoc/>
    public override int GetHashCode()
    {
        return _hashcode ??= HashCode.Combine(EqualityContract, Children, LeafExprEqualityComparer.Instance.GetHashCode(Expr));
    }

    /// <inheritdoc/>
    public override string ToString()
    {
        var str = string.Join(", ", Children.Select(x => x.Parent is null ? x.Id.ToString() : x.ToString()));
        return $"{Expr.GetType().Name} ({str})";
    }
}
