// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.IR
{
    /// <summary>
    /// Expression matutor.
    /// </summary>
    public abstract class ExprMutator : ExprVisitor<Expr, IRType>
    {
        /// <summary>
        /// for speedup the Mutator, If is Mutated we need MutateLeaf recursive.
        /// </summary>
        protected bool IsMutated = false;

        /// <inheritdoc/>
        public override Expr VisitLeaf(Call expr)
        {
            var nexpr = MutateLeaf(expr);
            if (!object.ReferenceEquals(expr, nexpr)) { IsMutated = true; return nexpr; }
            if (!IsMutated) return expr;
            return expr with
            {
                Target = Visit(expr.Target),
                Parameters = MutateArray(expr.Parameters, Visit),
            };
        }

        /// <inheritdoc/>
        public override Expr VisitLeaf(Const expr)
        {
            var nexpr = MutateLeaf(expr);
            if (!object.ReferenceEquals(expr, nexpr)) { IsMutated = true; return nexpr; }
            if (!IsMutated) return expr;
            return expr;
        }

        /// <inheritdoc/>
        public override Expr VisitLeaf(Function expr)
        {
            var nexpr = MutateLeaf(expr);
            if (!object.ReferenceEquals(expr, nexpr)) { IsMutated = true; return nexpr; }
            if (!IsMutated) return expr;
            return expr with
            {
                Body = Visit(expr.Body),
                Parameters = new(expr.Parameters.Select(Visit)),
            };
        }

        /// <inheritdoc/>
        public override Expr VisitLeaf(Op expr)
        {
            var nexpr = MutateLeaf(expr);
            if (!object.ReferenceEquals(expr, nexpr)) { IsMutated = true; return nexpr; }
            if (!IsMutated) return expr;
            return expr;
        }

        /// <inheritdoc/>
        public override Expr VisitLeaf(Tuple expr)
        {
            var nexpr = MutateLeaf(expr);
            if (!object.ReferenceEquals(expr, nexpr)) { IsMutated = true; return nexpr; }
            if (!IsMutated) return expr;
            return expr with
            {
                Fields = new(expr.Fields.Select(Visit)),
            };
        }

        /// <inheritdoc/>
        public override Expr VisitLeaf(Var expr)
        {
            var nexpr = MutateLeaf(expr);
            if (!object.ReferenceEquals(expr, nexpr)) { IsMutated = true; return nexpr; }
            if (!IsMutated) return expr;
            return expr;
        }

        /// <inheritdoc/>
        public override Expr VisitLeaf(TIR.IterVar expr)
        {
            var nexpr = MutateLeaf(expr);
            if (!object.ReferenceEquals(expr, nexpr)) { IsMutated = true; return nexpr; }
            if (!IsMutated) return expr;
            return expr with
            {
                Dom = MutateLeaf(expr.Dom),
                Value = Visit(expr.Value),
            };
        }

        /// <inheritdoc/>
        public override Expr VisitLeaf(TIR.Sequential expr)
        {
            var nexpr = MutateLeaf(expr);
            if (!object.ReferenceEquals(expr, nexpr)) { IsMutated = true; return nexpr; }
            if (!IsMutated) return expr;
            return expr with
            {
                Fields = MutateArray(expr.Fields, Visit),
            };
        }

        /// <inheritdoc/>
        public override Expr VisitLeaf(TIR.For expr)
        {
            var nexpr = MutateLeaf(expr);
            if (!object.ReferenceEquals(expr, nexpr)) { IsMutated = true; return nexpr; }
            if (!IsMutated) return expr;
            return expr with
            {
                LoopVar = (Var)Visit(expr.LoopVar),
                Dom = MutateLeaf(expr.Dom),
                Sequence = (TIR.Sequential)Visit(expr.Sequence),
            };
        }

        public override Expr VisitLeaf(TIR.Block expr)
        {
            var nexpr = MutateLeaf(expr);
            if (!object.ReferenceEquals(expr, nexpr)) { IsMutated = true; return nexpr; }
            if (!IsMutated) return expr;
            return expr with
            {
                // the block realize 
                InitSequence = (TIR.Sequential)Visit(expr.InitSequence),
                Predicate = Visit(expr.Predicate),
                IterVars = MutateArray(expr.IterVars, x => (TIR.IterVar)Visit(x)),

                // the block internal.
                Sequence = (TIR.Sequential)Visit(expr.Sequence),
                Reads = MutateArray(expr.Reads, MutateLeaf),
                Writes = MutateArray(expr.Writes, MutateLeaf),
            };
        }

        /// <inheritdoc/>
        public override Expr VisitLeaf(TIR.BufferStore expr)
        {
            var nexpr = MutateLeaf(expr);
            if (!object.ReferenceEquals(expr, nexpr)) { IsMutated = true; return nexpr; }
            if (!IsMutated) return expr;
            return expr with
            {
                Value = Visit(expr.Value),
                Indices = MutateArray(expr.Indices, Visit),
            };
        }

        /// <inheritdoc/>
        public override Expr VisitLeaf(TIR.BufferLoad expr)
        {
            var nexpr = MutateLeaf(expr);
            if (!object.ReferenceEquals(expr, nexpr)) { IsMutated = true; return nexpr; }
            if (!IsMutated) return expr;
            return expr with
            {
                Indices = MutateArray(expr.Indices, Visit),
            };
        }

        /// <summary>
        /// defulat mutate leaf is not mutate.
        /// </summary>
        /// <param name="expr"></param>
        /// <returns></returns>
        public virtual Expr DefaultMutateLeaf(Expr expr) => expr;

        /// <summary>
        /// mutate the call.
        /// </summary>
        /// <param name="expr"></param>
        /// <returns></returns>
        public virtual Expr MutateLeaf(Call expr) => DefaultMutateLeaf(expr);

        /// <summary>
        /// mutate the const.
        /// </summary>
        /// <param name="expr"></param>
        /// <returns></returns>
        public virtual Expr MutateLeaf(Const expr) => DefaultMutateLeaf(expr);

        /// <summary>
        /// mutate the function.
        /// </summary>
        /// <param name="expr"></param>
        /// <returns></returns>
        public virtual Expr MutateLeaf(Function expr) => DefaultMutateLeaf(expr);

        /// <summary>
        /// mutate the op.
        /// </summary>
        /// <param name="expr"></param>
        /// <returns></returns>
        public virtual Expr MutateLeaf(Op expr) => DefaultMutateLeaf(expr);

        /// <summary>
        /// mutate the tuple.
        /// </summary>
        /// <param name="expr"></param>
        /// <returns></returns>
        public virtual Expr MutateLeaf(Tuple expr) => DefaultMutateLeaf(expr);

        /// <summary>
        /// mutate the var.
        /// </summary>
        /// <param name="expr"></param>
        /// <returns></returns>
        public virtual Expr MutateLeaf(Var expr) => DefaultMutateLeaf(expr);

        /// <summary>
        /// mutate the itervar.
        /// </summary>
        /// <param name="expr"></param>
        /// <returns></returns>
        public virtual Expr MutateLeaf(TIR.IterVar expr) => DefaultMutateLeaf(expr);

        /// <summary>
        /// mutate the sequential.
        /// </summary>
        /// <param name="expr"></param>
        /// <returns></returns>
        public virtual Expr MutateLeaf(TIR.Sequential expr) => DefaultMutateLeaf(expr);

        /// <summary>
        /// mutate the for.
        /// </summary>
        /// <param name="expr"></param>
        /// <returns></returns>
        public virtual Expr MutateLeaf(TIR.For expr) => DefaultMutateLeaf(expr);

        /// <summary>
        /// mutate the block.
        /// </summary>
        /// <param name="expr"></param>
        /// <returns></returns>
        public virtual Expr MutateLeaf(TIR.Block expr) => DefaultMutateLeaf(expr);

        /// <summary>
        /// mutate the bufferstore.
        /// </summary>
        /// <param name="expr"></param>
        /// <returns></returns>
        public virtual Expr MutateLeaf(TIR.BufferStore expr) => DefaultMutateLeaf(expr);

        /// <summary>
        /// mutate the buffer load.
        /// </summary>
        /// <param name="expr"></param>
        /// <returns></returns>
        public virtual Expr MutateLeaf(TIR.BufferLoad expr) => DefaultMutateLeaf(expr);

        /// <summary>
        /// mutate irarray list.
        /// </summary>
        /// <typeparam name="TInput"></typeparam>
        /// <typeparam name="TResult"></typeparam>
        /// <param name="arrayList"></param>
        /// <param name="visitor"></param>
        /// <returns></returns>
        public virtual IRArrayList<TResult> MutateArray<TInput, TResult>(IRArrayList<TInput> arrayList, Func<TInput, TResult> visitor)
        {
            return new(arrayList.Select(visitor));
        }

        /// <summary>
        /// Mutate IRArray.
        /// </summary>
        /// <typeparam name="TInput"></typeparam>
        /// <typeparam name="TResult"></typeparam>
        /// <param name="array"></param>
        /// <param name="visitor"></param>
        /// <returns></returns>
        public virtual IRArray<TResult> MutateArray<TInput, TResult>(IRArray<TInput> array, Func<TInput, TResult> visitor)
        {
            return new(array.Select(visitor));
        }

        /// <summary>
        /// mutate range.
        /// </summary>
        /// <param name="range"></param>
        /// <returns></returns>
        public virtual TIR.Range MutateLeaf(TIR.Range range)
        {
            if (!IsMutated) return range;
            return range with
            {
                Min = Visit(range.Min),
                Max = Visit(range.Max),
            };
        }

        /// <summary>
        /// mutate the buffer region.
        /// </summary>
        /// <param name="region"></param>
        /// <returns></returns>
        public virtual TIR.BufferRegion MutateLeaf(TIR.BufferRegion region)
        {
            if (!IsMutated) return region;
            return region with
            {
                Region = MutateArray(region.Region, MutateLeaf),
            };
        }
    }
}
