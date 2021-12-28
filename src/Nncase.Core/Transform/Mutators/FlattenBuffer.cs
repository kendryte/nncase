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
        public override Expr Visit(Block expr)
        {
            if (expr.IterVars.Count != 0)
                throw new InvalidOperationException("Non-opaque blocks are not allowed in FlattenBuffer. Please call pass ConvertBlocksToOpaque before.");
            // 1. Visit the body
            var nbody = Visit(expr.Body);
            IRArrayList<BufferRegion> nreads = new(expr.Reads.Select(VisitBufferRegion));
            IRArrayList<BufferRegion> nwrites = new(expr.Writes.Select(VisitBufferRegion));
            IRArrayList<TIR.Buffer> nbufs = new(expr.AllocBuffers.Select(VisitBuffer));
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
        public override Expr Visit(For expr)
        {
            // Step 1. Update unit loop info.
            var min = Visit(expr.Dom.Min);
            var max = Visit(expr.Dom.Max);
            // Step 2. Visit recursively
            var body = Visit(expr.Body);
            return new For(expr.LoopVar, new(min, max), expr.Mode, (Sequential)body);
        }

        /// <inheritdoc/>
        public override Expr Visit(BufferLoad expr)
        {
            var load = (BufferLoad)base.Visit(expr);
            return expr.Buffer.VLoad(load.Indices);
        }

        /// <inheritdoc/>
        public override Expr Visit(BufferStore expr)
        {
            var store = (BufferStore)base.Visit(expr);
            return expr.Buffer.VStore(store.Indices, store.Value);
        }
    }
}