// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.Utilities;

namespace Nncase.TIR;

/// <summary>
/// Buffer load node.
/// </summary>
public sealed class BufferLoad : Expr
{
    public BufferLoad(PhysicalBuffer buffer, ReadOnlySpan<Expr> indices)
        : base(ArrayUtility.Concat(buffer, indices))
    {
    }

    /// <summary>
    /// Gets the buffer to be loaded.
    /// </summary>
    public PhysicalBuffer Buffer => (PhysicalBuffer)Operands[0];

    /// <summary>
    /// Gets the buffer indices.
    /// </summary>
    public ReadOnlySpan<Expr> Indices => Operands.Slice(1);

    /// <inheritdoc/>
    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context)
        => functor.VisitBufferLoad(this, context);

    public BufferLoad With(PhysicalBuffer? buffer = null, Expr[]? indices = null)
        => new BufferLoad(buffer ?? Buffer, indices ?? Indices);
}
