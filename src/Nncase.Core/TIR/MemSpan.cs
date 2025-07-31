// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase;
using Nncase.IR;

namespace Nncase.TIR;

public sealed class MemSpan : BaseExpr
{
    public MemSpan(PhysicalBuffer buffer, Dimension start, Dimension size)
        : base([buffer, start, size])
    {
    }

    public MemSpan(PhysicalBuffer buffer)
        : this(buffer, 0, buffer.Size)
    {
    }

    /// <summary>
    /// Gets the buffer.
    /// </summary>
    public PhysicalBuffer Buffer => (PhysicalBuffer)Operands[0];

    /// <summary>
    /// Gets the start.
    /// </summary>
    public Dimension Start => (Dimension)Operands[1];

    /// <summary>
    /// Gets the size of bytes.
    /// </summary>
    public Dimension Size => (Dimension)Operands[2];

    /// <inheritdoc/>
    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context)
        => functor.VisitMemSpan(this, context);

    public MemSpan With(PhysicalBuffer? buffer = null, Dimension? start = null, Dimension? size = null) =>
        new(buffer ?? Buffer, start ?? Start, size ?? Size);
}
