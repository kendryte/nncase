// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

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
    public class LowerBlockInit : ExprMutator
    {
        /// <inheritdoc/>
        public override Expr VisitLeaf(Block expr)
        {
            if (expr.InitBody.Count == 0)
            {
                return base.Visit(expr);
            }

            var initbody = Lowering(expr.InitBody, expr.IterVars);
            var body = Visit(expr.Body);
            return expr with
            {
                InitBody = new(),
                Body = Sequential.Flatten(initbody, body),
            };
        }

        Expr Lowering(Sequential init, IRArrayList<IterVar> iterVars)
        {
            List<Expr> conds = new();
            foreach (var iterVar in iterVars)
            {
                if (iterVar.Mode == IterationMode.CommReduce)
                {
                    conds.Append(IR.F.Math.Equal(iterVar, iterVar.Dom.Min));
                }
            }

            if (conds.Count == 0)
            {
                return init;
            }

            var cond = conds[0];
            foreach (var i in Enumerable.Range(1, conds.Count - 1))
            {
                cond = IR.F.Math.LogicalAnd(cond, conds[i]);
            }

            return new IfThenElse(cond, init);
        }
    }
}