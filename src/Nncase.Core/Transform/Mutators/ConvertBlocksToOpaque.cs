using System.Linq;
using System.IO;
using System.Collections.Generic;
using System;
using Nncase.TIR;
using Nncase.IR;

namespace Nncase.Transform.Mutator
{
    /// <summary>
    /// Substitute all the block vars with the PrimExprs they are bound to, indicated by the corresponding iter_values in BlockRealize, for opaque blocks by removing all . the iter_values in BlockRealize and iter_vars in Block.
    /// </summary>
    public class ConvertBlocksToOpaque : ExprMutator
    {
        /// <inheritdoc/>
        public override Expr MutateLeaf(IterVar expr)
        { 
            return expr.Value;
        }

        /// <inheritdoc/>
        public override Expr MutateLeaf(Block expr)
        {
            return expr with
            {
                // the block realize 
                InitBody = (TIR.Sequential)Visit(expr.InitBody),
                Predicate = Visit(expr.Predicate),
                IterVars = new(),
                // the block internal.
                Body = (TIR.Sequential)Visit(expr.Body),
                Reads = MutateArray(expr.Reads, MutateLeaf),
                Writes = MutateArray(expr.Writes, MutateLeaf)
            };
        }
    }
}