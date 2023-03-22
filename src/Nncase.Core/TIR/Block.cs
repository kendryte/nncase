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
/// A block is a basic schedule unit in TIR.
/// <remarks>
/// Block's body is parameterized by iter vars.
/// </remarks>
/// <code>
///   with T.block(name):
///   v0 = T.axis.S(domain, value0)
///   v1 = T.axis.R(domain, value1)
///   ...
///   T.reads([buffer0[start:end, ...], ...])
///   T.writes([buffer1[start:end, ...], ...])
///   T.where(predicate)
///   buffer2 = T.alloc_buffer(shape, dtype)
///   buffer3 = T.match_buffer(source_buffer[start:end, ...])
///   T.attr({attr_key: attr_value, ...})
///   with T.init():
///      init body
///    body
/// </code>
/// </summary>
public sealed class Block : Expr
{
    private readonly int _iterVarsCount;
    private readonly int _readsCount;
    private readonly int _writesCount;
    private readonly int _allocBuffersCount;

    public Block(string name, Sequential body, Sequential initBody, ReadOnlySpan<IterVar> iterVars, ReadOnlySpan<BufferRegion> reads, ReadOnlySpan<BufferRegion> writes, ReadOnlySpan<Buffer> allocBuffers, Expr predicate)

        // TODO: Optimize allocates
        : base(new Expr[] { body, initBody }.Concat(iterVars.ToArray()).Concat(reads.ToArray()).Concat(writes.ToArray()).Concat(allocBuffers.ToArray()).Append(predicate).ToArray())
    {
        Name = name;
        _iterVarsCount = iterVars.Length;
        _readsCount = reads.Length;
        _writesCount = writes.Length;
        _allocBuffersCount = allocBuffers.Length;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="Block"/> class.
    /// </summary>
    public Block(string name)
        : this(name, new(), new(), Array.Empty<IterVar>(), Array.Empty<BufferRegion>(), Array.Empty<BufferRegion>(), Array.Empty<Buffer>(), true)
    {
    }

    /// <summary>
    /// Gets the name_hint of the block.
    /// </summary>
    public string Name { get; }

    /// <summary>
    /// Gets block body.
    /// </summary>
    public Sequential Body => (Sequential)Operands[0];

    /// <summary>
    /// Gets the Block init statement.
    /// </summary>
    public Sequential InitBody => (Sequential)Operands[1];

    /// <summary>
    /// Gets the List Exprs contain the IterVars.
    /// </summary>
    public ReadOnlySpan<IterVar> IterVars => SpanUtility.UnsafeCast<Expr, IterVar>(Operands.Slice(2, _iterVarsCount));

    /// <summary>
    /// Gets the read buffer regions of the block.
    /// </summary>
    public ReadOnlySpan<BufferRegion> Reads => SpanUtility.UnsafeCast<Expr, BufferRegion>(Operands.Slice(2 + _iterVarsCount, _readsCount));

    /// <summary>
    /// Gets the write buffer regions of the block.
    /// </summary>
    public ReadOnlySpan<BufferRegion> Writes => SpanUtility.UnsafeCast<Expr, BufferRegion>(Operands.Slice(2 + _iterVarsCount + _readsCount, _writesCount));

    /// <summary>
    /// Gets the buffer allocated in the block.
    /// </summary>
    public ReadOnlySpan<Buffer> AllocBuffers => SpanUtility.UnsafeCast<Expr, Buffer>(Operands.Slice(2 + _iterVarsCount + _readsCount + _writesCount, _allocBuffersCount));

    /// <summary>
    /// Gets the predicate of the block realization, the block will only be executed when the predicate is true.
    /// </summary>
    public Expr Predicate => Operands[2 + _iterVarsCount + _readsCount + _writesCount + _allocBuffersCount];

    /// <inheritdoc/>
    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context)
        => functor.VisitBlock(this, context);

    public Block With(string? name = null, Sequential? body = null, Sequential? initBody = null, IterVar[]? iterVars = null, BufferRegion[]? reads = null, BufferRegion[]? writes = null, Buffer[]? allocBuffers = null, Expr? predicate = null)
        => new Block(name ?? Name, body ?? Body, initBody ?? InitBody, iterVars ?? IterVars, reads ?? Reads, writes ?? Writes, allocBuffers ?? AllocBuffers, predicate ?? Predicate);
}
