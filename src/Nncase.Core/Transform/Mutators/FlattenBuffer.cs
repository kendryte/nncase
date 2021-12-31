using System.Linq;
using System.IO;
using System.Collections.Generic;
using System;
using Nncase.TIR;
using Nncase.IR;

namespace Nncase.Transform.Mutator
{
    /// <summary>
    /// Flatten the multi-dimensional BufferLoad and BufferStore to single dimensional Load/Store. Also remove Block to ensure that the flattened TIR can not be scheduled again.
    /// </summary>
    public class FlattenBuffer : ExprMutator
    {
        /// <inheritdoc/>
        public override Expr MutateLeaf(Block expr)
        {
            if (expr.IterVars.Count != 0)
                throw new InvalidOperationException("Non-opaque blocks are not allowed in FlattenBuffer. Please call pass ConvertBlocksToOpaque before.");
            // 1. Visit the body
            var nbody = Visit(expr.Body);
            IRArrayList<BufferRegion> nreads = new(expr.Reads.Select(MutateLeaf));
            IRArrayList<BufferRegion> nwrites = new(expr.Writes.Select(MutateLeaf));
            var npredicate = Visit(expr.Predicate);
            if (npredicate != (Const)1)
            {
                nbody = new TIR.IfThenElse(npredicate, nbody);
            }
            // Step 3. Handle allocations in reverse order
            // TODO add the alloc buffers.
            // for (size_t i = new_block->alloc_buffers.size(); i > 0; --i) {
            //   const Buffer& buffer = new_block->alloc_buffers[i - 1];
            //   body = MakeAllocStmt(buffer, std::move(body));
            // }
            return nbody;
        }

        /// <inheritdoc/>
        public override Expr MutateLeaf(BufferLoad expr)
        {
            return expr.Buffer.VLoad(MutateArray(expr.Indices, Visit));
        }

        /// <inheritdoc/>
        public override Expr MutateLeaf(BufferStore expr)
        {
            return expr.Buffer.VStore(MutateArray(expr.Indices, Visit), Visit(expr.Value));
        }
    }
}