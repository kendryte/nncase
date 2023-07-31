// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reactive;
using Nncase.IR;
using Nncase.TIR;

namespace Nncase.Passes.Mutators;

/// <summary>
/// Flatten the multi-dimensional BufferLoad and BufferStore to single dimensional Load/Store.
/// </summary>
public sealed class FlattenBuffer : ExprRewriter
{
    /// <inheritdoc/>
    protected override Expr RewriteLeafBlock(Block expr)
    {
        // TODO: put the unfold block into this.
        if (expr.Predicate is TensorConst tc && tc.Value.ToScalar<bool>() == true)
        {
            return expr.Body;
        }

        return T.Nop();
    }

    /// <inheritdoc/>
    protected override Expr RewriteLeafCall(Call expr)
    {
        if (expr.Target is IR.Buffers.BufferLoad)
        {
            var indices = (IR.Tuple)expr[IR.Buffers.BufferLoad.Indices];
            var input = (TIR.Buffer)expr[IR.Buffers.BufferLoad.Input];
            return T.Load(input.MemSpan, Enumerable.Range(0, indices.Count).Aggregate((Expr)0, (acc, i) => acc + (input.Strides[i] * indices[i])));
        }
        else if (expr.Target is IR.Buffers.BufferStore)
        {
            var indices = (IR.Tuple)expr[IR.Buffers.BufferStore.Indices];
            var input = (TIR.Buffer)expr[IR.Buffers.BufferStore.Input];
            return T.Store(input.MemSpan, Enumerable.Range(0, indices.Count).Aggregate((Expr)0, (acc, i) => acc + (input.Strides[i] * indices[i])), expr[IR.Buffers.BufferStore.Value]);
        }
        else if (expr.Target is IR.Buffers.MatchBuffer && expr.Arguments[0] is TIR.Buffer { MemSpan: { Start: (Const or Var) } })
        {
            // remove the all fixed match operation.
            return T.Nop();
        }

        return expr;
    }
}
