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
#if false
    /// <summary>
    /// Create a BufferRegion which is full region of the given buffer.
    /// NOTE because of the each backend has different addr calc logic.
    /// </summary>
    /// <param name="buffer">The buffer to generate full BufferRegion.</param>
    /// <returns>The BufferRegion which covers all region of the given buffer.</returns>
    public static BufferRegion All(PhysicalBuffer buffer) => new BufferRegion(buffer, new(buffer.Dimensions.Select(extent => new Range(0, extent, 1))));

    /// <summary>
    /// Get the RegionSize.
    /// </summary>
    public Expr[] RegionSize => Region.Select(r => r.Stop - r.Start).ToArray();

    /// <summary>
    /// Get padding at the dim.
    /// </summary>
    /// <param name="dim"></param>
    /// <returns></returns>
    public (Expr Before, Expr After) Padding(int dim) => (IR.F.Math.Max(-Region[dim].Start, 0), IR.F.Math.Max(Region[dim].Stop - Buffer.Dimensions[dim], 0));
#endif

    public BufferRegion(Buffer buffer, ReadOnlySpan<Range> region)
        : base(ArrayUtility.Concat(buffer, SpanUtility.UnsafeCast<Range, Expr>(region)))
    {
    }

    /// <summary>
    /// Gets the buffer of the buffer region.
    /// </summary>
    public Buffer Buffer => (Buffer)Operands[0];

    /// <summary>
    /// Gets the region array of the buffer region.
    /// </summary>
    public ReadOnlySpan<Range> Region => SpanUtility.UnsafeCast<Expr, Range>(Operands.Slice(1));

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
                      Call { Target: IR.Math.Unary { UnaryOp: UnaryOp.Neg } } => throw new NotSupportedException("Neg Region!"),

                      // else return the origin range.
                      _ => tp.Second,
                  }).ToArray()));
    }

    /// <inheritdoc/>
    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context)
        => functor.VisitBufferRegion(this, context);

    public BufferRegion With(Buffer? buffer, Range[]? region = null)
        => new BufferRegion(buffer ?? Buffer, region ?? Region);
}
