// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reactive;
using Nncase.IR;
using Nncase.TIR;

namespace Nncase.TIR;

public class Scheduler
{
    public Scheduler(Function entry)
    {
        Entry = entry;
    }

    public Function Entry { get; set; }

    /// <summary>
    /// get the block instance by name.
    /// </summary>
    public Block GetBlock(string blockName)
    {
        Block? targetBlock = null;
        void CollectBlock(Expr expr)
        {
            if (expr is Block b && b.Name == blockName)
            {
                if (targetBlock is null)
                {
                    targetBlock = b;
                }
                else
                {
                    throw new InvalidOperationException($"Find The Duplicate Block {blockName}!");
                }
            }
        }

        var collector = new ExprCollector(CollectBlock);
        collector.Visit(Entry);
        if (targetBlock is null)
        {
            throw new InvalidOperationException($"Can't Find The Block Name {blockName}!");
        }

        return targetBlock;
    }

    /// <summary>
    /// recursive find block direct parent loop!.
    /// </summary>
    /// <returns> the loops. </returns>
    public For[] GetLoops(Block block)
    {
        List<For> targetLoops = new();
        Expr child = block;
        void CollectLoops(Expr expr)
        {
            if (expr is For parent && object.ReferenceEquals(parent.Body[0], child))
            {
                targetLoops.Insert(0, parent);
                child = parent;
            }
        }

        var collector = new ExprCollector(CollectLoops);
        collector.Visit(Entry);
        return targetLoops.ToArray();
    }

    public For[] Split(For loop, params Expr[] factors)
    {
        // step 1. check the arguments
        if (loop.Domain.Start != (Const)0)
        {
            throw new NotImplementedException("Loop Not Start With Zero");
        }

        Expr tolLength = 1;
        foreach (var factor in factors)
        {
            tolLength = tolLength * factor;
        } // TODO add assert total == (loop.Dom.Max - loop.Dom.Min)

        // Step 2. Replace all occurrences of the original loop var with new variables
        Expr total = 1, substitute = 0;
        var newloopVars = new Var[factors.Length];
        foreach (var i in Enumerable.Range(0, factors.Length))
        {
            var loopVar = new Var(TensorType.Scalar(DataTypes.Int32));
            substitute = (substitute * factors[i]) + loopVar;
            newloopVars[i] = loopVar;
        }

        Dictionary<Block, Block> opaque_block_reuse = new(); // TODO the opaque_block_reuse for what?
        Sequential nbody = loop.Body;

        // Step 3. create new for loop.
        var nFor = new For[factors.Length];

        // nbody = (Sequential)new Passes.Mutators.SubstituteVarAndCollectOpaqueBlock(v => v == loop.LoopVar ? substitute : v, opaque_block_reuse).Rewrite(nbody);
        // for (int i = factors.Length - 1; i >= 0; i--)
        // {
        //     var @for = new For(newloopVars[i], (0, factors[i]), LoopMode.Serial, nbody);
        //     nbody = T.Sequential(@for);
        //     nFor[i] = @for;
        // }

        // // Setp 4. update the function
        // Entry = (Function)new Passes.Mutators.Substitutor(expr => object.ReferenceEquals(expr, loop) ? nFor[0] : null).Rewrite(Entry);
        return nFor;
    }

    private sealed class ExprCollector : ExprWalker
    {
        private readonly Action<Expr> _collectFunc;

        public ExprCollector(Action<Expr> func)
        {
            _collectFunc = func;
        }

        protected override Unit DefaultVisitLeaf(Expr expr)
        {
            _collectFunc(expr);
            return default;
        }
    }
}
