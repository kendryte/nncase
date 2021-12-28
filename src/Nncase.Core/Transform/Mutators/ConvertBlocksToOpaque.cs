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
        public override Expr Visit(IterVar expr)
        {
            return expr.Value;
        }

        public override Expr Visit(Block expr)
        {
            return expr with
            {
                Body = (TIR.Sequential)Visit(expr.Body),
                InitBody = (TIR.Sequential)Visit(expr.InitBody),
                IterVars = new(),
                Reads = new(expr.Reads.Select(VisitBufferRegion)),
                Writes = new(expr.Writes.Select(VisitBufferRegion)),
                AllocBuffers = new(expr.AllocBuffers.Select(VisitBuffer)),
                Predicate = Visit(expr.Predicate)
            };
        }
    }
}