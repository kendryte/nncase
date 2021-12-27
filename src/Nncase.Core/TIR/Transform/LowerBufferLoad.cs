using System.Linq;
using System.IO;
using System.Collections.Generic;
using System;
using Nncase.TIR;
using Nncase.IR;

namespace Nncase.Transform.TIRPass
{
    public class ConvertBlocksToOpaquePass : FunctionPass
    {
        public ConvertBlocksToOpaquePass() : base("LowerBufferLoad") { }

        /// <inheritdoc/>
        protected override Function RunCore(Function function, RunPassOptions options)
        {
            return (Function)new ConvertBlocksToOpaque().Visit(function);
        }
    }

    class ConvertBlocksToOpaque : ExprMatutor
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
                IterVarBinds = new(),
                Reads = new(expr.Reads.Select(VisitBufferRegion)),
                Writes = new(expr.Writes.Select(VisitBufferRegion)),
                AllocBuffers = new(expr.AllocBuffers.Select(VisitBuffer)),
                Predicate = Visit(expr.Predicate)
            };
        }
    }
}