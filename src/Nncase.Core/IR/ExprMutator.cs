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
        /// for speedup the Mutator, If is Mutated we need Mutate recursive.
        /// </summary>
        protected bool IsMutated = false;

        /// <summary>
        /// the default visit the  Original leaf expr, we can hook it.
        /// </summary>
        /// <param name="expr"></param>
        /// <returns></returns>
        public virtual Expr DefaultVisitLeafOrigin(Expr expr) => expr;

        /// <inheritdoc/>
        public override Expr VisitLeaf(Call expr)
        {
            if (!IsMutated) return DefaultVisitLeafOrigin(expr);
            return expr with
            {
                Target = Visit(expr.Target),
                Parameters = Mutate(expr.Parameters, Visit)
            };
        }

        /// <inheritdoc/>
        public override Expr VisitLeaf(Const expr)
        {
            if (!IsMutated) return DefaultVisitLeafOrigin(expr);
            return expr;
        }

        /// <inheritdoc/>
        public override Expr VisitLeaf(Function expr)
        {
            if (!IsMutated) return DefaultVisitLeafOrigin(expr);
            return expr with
            {
                Body = Visit(expr.Body),
                Parameters = new(expr.Parameters.Select(Visit)),
            };
        }

        /// <inheritdoc/>
        public override Expr VisitLeaf(Op expr)
        {
            if (!IsMutated) return DefaultVisitLeafOrigin(expr);
            return expr;
        }

        /// <inheritdoc/>
        public override Expr VisitLeaf(Tuple expr)
        {
            if (!IsMutated) return DefaultVisitLeafOrigin(expr);
            return expr with
            {
                Fields = new(expr.Fields.Select(Visit)),
            };
        }

        /// <inheritdoc/>
        public override Expr VisitLeaf(Var expr)
        {
            if (!IsMutated) return DefaultVisitLeafOrigin(expr);
            return expr;
        }

        /// <inheritdoc/>
        public override Expr VisitLeaf(TIR.IterVar expr)
        {
            if (!IsMutated) return DefaultVisitLeafOrigin(expr);
            return expr with
            {
                Dom = Mutate(expr.Dom),
                Value = Visit(expr.Value)
            };
        }

        /// <inheritdoc/>
        public override Expr VisitLeaf(TIR.Sequential expr)
        {
            if (!IsMutated) return DefaultVisitLeafOrigin(expr);
            return expr with
            {
                Fields = MutateArray(expr.Fields, Visit),
            };
        }

        /// <inheritdoc/>
        public override Expr VisitLeaf(TIR.For expr)
        {
            if (!IsMutated) return DefaultVisitLeafOrigin(expr);
            return expr with
            {
                LoopVar = (Var)Visit(expr.LoopVar),
                Dom = Mutate(expr.Dom),
                Body = (TIR.Sequential)Visit(expr.Body),
            };
        }

        public override Expr VisitLeaf(TIR.Block expr)
        {
            if (!IsMutated) return DefaultVisitLeafOrigin(expr);
            return expr with
            {
                // the block realize 
                InitBody = (TIR.Sequential)Visit(expr.InitBody),
                Predicate = Visit(expr.Predicate),
                IterVars = MutateArray(expr.IterVars, x => (TIR.IterVar)Visit(x)),
                // the block internal.
                Body = (TIR.Sequential)Visit(expr.Body),
                Reads = MutateArray(expr.Reads, Mutate),
                Writes = MutateArray(expr.Writes, Mutate)
            };
        }

        /// <inheritdoc/>
        public override Expr VisitLeaf(TIR.BufferStore expr)
        {
            if (!IsMutated) return DefaultVisitLeafOrigin(expr);
            return expr with
            {
                Value = Visit(expr.Value),
                Indices = Mutate(expr.Indices, Visit),
            };
        }

        /// <inheritdoc/>
        public override Expr VisitLeaf(TIR.BufferLoad expr)
        {
            if (!IsMutated) return DefaultVisitLeafOrigin(expr);
            return expr with
            {
                Indices = Mutate(expr.Indices, Visit)
            };
        }

        public virtual IRArrayList<TResult> MutateArray<TInput, TResult>(IRArrayList<TInput> arrayList, Func<TInput, TResult> visitor)
        {
            return new(arrayList.Select(visitor));
        }

        public virtual IRArray<TResult> Mutate<TInput, TResult>(IRArray<TInput> array, Func<TInput, TResult> visitor)
        {
            return new(array.Select(visitor));
        }

        /// <summary>
        /// visit range.
        /// </summary>
        /// <param name="range"></param>
        /// <returns></returns>
        public virtual TIR.Range Mutate(TIR.Range range)
        {
            if (!IsMutated) return range;
            return range with
            {
                Min = Visit(range.Min),
                Max = Visit(range.Max)
            };
        }

        /// <summary>
        /// visit the buffer region
        /// </summary>
        /// <param name="region"></param>
        /// <returns></returns>
        public virtual TIR.BufferRegion Mutate(TIR.BufferRegion region)
        {
            if (!IsMutated) return region;
            return region with
            {
                Region = Mutate(region.Region, Mutate),
            };
        }
    }
}
