using Nncase.IR;
using Nncase.TIR;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Nncase.TIR
{

    class Substitutor : ExprMutator
    {
        Func<Expr, Expr?> Maper;

        public Substitutor(Func<Expr, Expr?> maper)
        {
            Maper = maper;
        }

        /// <inheritdoc/>
        public override Expr DefaultVisitLeafOrigin(Expr expr)
        {
            var mexpr = Maper(expr);
            if (mexpr is not null) { IsMutated = true; return mexpr; }
            return expr;
        }
    }
    /// <summary>
    /// Substitute vars and collect the reuse mapping of opaque blocks
    /// </summary>
    class SubstituteVarAndCollectOpaqueBlock : ExprMutator
    {

        Func<Var, Expr?> VarMaper;

        readonly Dictionary<Block, Block> OpaqueBlocks;
        public SubstituteVarAndCollectOpaqueBlock(Func<Var, Expr?> varMaper,
                                              Dictionary<Block, Block> opaque_blocks)
        {
            VarMaper = varMaper;
            OpaqueBlocks = opaque_blocks;
        }

        /// <inheritdoc/>
        public override Expr VisitLeaf(Var expr)
        {
            if (VarMaper(expr) is var nexpr && nexpr is not null)
            {
                IsMutated = true;
                return nexpr;
            }
            return expr;
        }

        /// <inheritdoc/>
        public override Expr VisitLeaf(Block expr)
        {
            var nblock = (Block)base.VisitLeaf(expr);
            if (nblock.IterVars.Count == 0)
            {
                OpaqueBlocks.Add(expr, nblock);
            }
            return nblock;
        }
    };

    class TIRCollector : ExprVisitor<bool, bool>
    {
        Action<Expr> CollectFunc;
        public TIRCollector(Action<Expr> func)
        {
            CollectFunc = func;
        }
        /// <inheritdoc/>
        public override bool DefaultVisitLeaf(Expr expr)
        {
            CollectFunc(expr);
            return true;
        }
    }

    public class Scheduler
    {
        public Function Entry;
        public Scheduler(Function entry)
        {
            Entry = entry;
        }

        /// <summary>
        /// get the block instance by name.
        /// </summary>
        /// <param name="blockName"></param>
        /// <returns></returns>
        /// <exception cref="InvalidOperationException"></exception>
        public Block GetBlock(string blockName)
        {
            Block? TargetBlock = null;
            void collectBlock(Expr expr)
            {
                if (expr is Block b && b.Name == blockName)
                {
                    if (TargetBlock is null)
                        TargetBlock = b;
                    else
                        throw new InvalidOperationException($"Find The Duplicate Block {blockName}!");
                }
            }
            var collector = new TIRCollector(collectBlock);
            collector.Visit(Entry);
            if (TargetBlock is null) throw new InvalidOperationException($"Can't Find The Block Name {blockName}!");
            return TargetBlock;
        }

        /// <summary>
        /// recursive find block direct parent loop!
        /// </summary>
        /// <param name="block"></param>
        /// <returns> the loops. </returns>
        public For[] GetLoops(Block block)
        {
            List<For> targetLoops = new();
            Expr child = block;
            void collectLoops(Expr expr)
            {
                if (expr is For parent && object.ReferenceEquals(parent.Body[0], child))
                {
                    targetLoops.Insert(0, parent);
                    child = parent;
                };
            }
            var collector = new TIRCollector(collectLoops);
            collector.Visit(Entry);
            return targetLoops.ToArray();
        }

        public For[] Split(For loop, params Expr[] factors)
        {
            // step 1. check the arguments
            if (loop.Dom.Min != (Const)0) { throw new NotImplementedException("Loop Not Start With Zero"); }
            Expr tolLength = 1;
            foreach (var factor in factors) { tolLength = tolLength * factor; } // TODO add assert total == (loop.Dom.Max - loop.Dom.Min)

            // Step 2. Replace all occurrences of the original loop var with new variables
            Expr total = 1, substitute = 0;
            var newloopVars = new Var[factors.Length];
            foreach (var i in Enumerable.Range(0, factors.Length))
            {
                var loopVar = new Var(TensorType.Scalar(DataType.Int32));
                substitute = substitute * factors[i] + loopVar;
                newloopVars[i] = loopVar;
            }
            Dictionary<Block, Block> opaque_block_reuse = new(); // TODO the opaque_block_reuse for what?
            Sequential nbody = loop.Body;
            // Step 3. create new for loop.
            var nFor = new For[factors.Length];
            nbody = (Sequential)new SubstituteVarAndCollectOpaqueBlock(v => v == loop.LoopVar ? substitute : v, opaque_block_reuse).Visit(nbody);
            for (int i = factors.Length - 1; i >= 0; i--)
            {
                var @for = new For(newloopVars[i], (0, factors[i]), ForMode.Serial, nbody);
                nbody = new Sequential() { @for };
                nFor[i] = @for;
            }
            // Setp 4. update the function
            Entry = (Function)new Substitutor(expr => object.ReferenceEquals(expr, loop) ? nFor[0] : null).Visit(Entry);
            return nFor;
        }
    }
}
