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
/// Buffer store node.
/// </summary>
public sealed class BufferStore : Expr
{
    private readonly int _indicesCount;

    public BufferStore(PhysicalBuffer buffer, ReadOnlySpan<Expr> indices, Expr value)
        : base(new Expr[] { buffer }.Concat(indices.ToArray()).Append(value).ToArray())
    {
        _indicesCount = indices.Length;
    }

    /// <summary>
    /// Gets the buffer.
    /// </summary>
    public PhysicalBuffer Buffer => (PhysicalBuffer)Operands[0];

    /// <summary>
    /// Gets the value we to be stored.
    /// </summary>
    public ReadOnlySpan<Expr> Indices => Operands[1.._indicesCount];

    /// <summary>
    /// Gets the indices location to be stored.
    /// </summary>
    public Expr Value => Operands[_indicesCount + 1];

    /// <inheritdoc/>
    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context)
        => functor.VisitBufferStore(this, context);

    public BufferStore With(PhysicalBuffer? buffer = null, Expr[]? indices = null, Expr? value = null)
        => new BufferStore(buffer ?? Buffer, indices ?? Indices, value ?? Value);
}
