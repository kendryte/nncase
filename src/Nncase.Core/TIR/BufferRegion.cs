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
/// Representing the region of multi-dimensional buffer access.
/// NOTE the region can be negative, we can use negative calc the padding.
/// </summary>
public sealed class BufferRegion : Expr
{
    public BufferRegion(Expr buffer, ReadOnlySpan<Range> region)
        : base(ArrayUtility.Concat(buffer, SpanUtility.UnsafeCast<Range, BaseExpr>(region)))
    {
    }

    /// <summary>
    /// Gets the buffer of the buffer region.
    /// </summary>
    public Expr Buffer => (Expr)Operands[0];

    /// <summary>
    /// Gets the region array of the buffer region.
    /// </summary>
    public ReadOnlySpan<Range> Region => SpanUtility.UnsafeCast<BaseExpr, Range>(Operands.Slice(1));

    /// <summary>
    /// Gets new buffer region.
    /// </summary>
    public BufferRegion this[params Range[] ranges]
    {
        get => new(Buffer, new(Region.ToArray().Zip(ranges).Select(
            tp => tp.Second.Equals(System.Range.All) ?
                  tp.First :
                  tp.Second.Stop switch
                  {
                      // if stop is neg, add the shape
                      Dimension d when d.Metadata.Range?.Min < 0 => throw new NotSupportedException("Neg Region!"),

                      // else return the origin range.
                      _ => tp.Second,
                  }).ToArray()));
    }

    /// <inheritdoc/>
    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context)
        => functor.VisitBufferRegion(this, context);

    public BufferRegion With(Expr? buffer, Range[]? region = null)
        => new BufferRegion(buffer ?? Buffer, region ?? Region);
}
