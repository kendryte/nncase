using System.IO;
using System.Collections.Generic;
using System;
using Nncase.IR;
using Nncase.TIR;

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
        // public override Expr Visit(Var expr)
        // {
        // }

        public override Expr Visit(Block expr)
        {
            var newblk = (Block)base.Visit(expr);
            newblk.IterVarPairs.Clear();
            return newblk;
        }
    }
}