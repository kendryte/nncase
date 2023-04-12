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
/// Flatten the multi-dimensional BufferLoad and BufferStore to single dimensional Load/Store. Also remove Block to ensure that the flattened TIR can not be scheduled again.
/// </summary>
public sealed class FlattenBuffer : ExprRewriter
{
    /// <inheritdoc/>
    protected override Expr RewriteLeafBlock(Block expr)
    {
        if (!expr.IterVars.IsEmpty)
        {
            throw new InvalidOperationException("Non-opaque blocks are not allowed in FlattenBuffer. Please call pass ConvertBlocksToOpaque before.");
        }

        // 1. Visit the body
        var predicate = expr.Predicate;
        if (predicate is TensorConst { Value: { Length: 1 } t }
            && t.ToScalar<bool>())
        {
            return expr.Body;
        }
        else
        {
            return new IfThenElse(predicate, expr.Body);
        }

        // Step 3. Handle allocations in reverse order
        // TODO add the alloc buffers.
        // for (size_t i = new_block->alloc_buffers.size(); i > 0; --i) {
        //   const Buffer& buffer = new_block->alloc_buffers[i - 1];
        //   body = MakeAllocStmt(buffer, std::move(body));
        // }
    }
}
