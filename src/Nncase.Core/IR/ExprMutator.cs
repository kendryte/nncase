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
    public abstract class ExprMutator : ExprFunctor<Expr, IRType>
    {
        RecordRefComparer<Expr> comparer = new();

        /// <inheritdoc/>
        public override Expr Visit(Call expr)
        {
            return expr with
            {
                Target = Visit(expr.Target),
                Parameters = new(expr.Parameters.Select(Visit)),
            };
        }

        /// <inheritdoc/>
        public override Expr Visit(Const expr)
        {
            return expr;
        }

        /// <inheritdoc/>
        public override Expr Visit(Function expr)
        {
            return expr with
            {
                Body = Visit(expr.Body),
                Parameters = new(expr.Parameters.Select(Visit)),
            };
        }

        /// <inheritdoc/>
        public override Expr Visit(Op expr)
        {
            return expr;
        }

        /// <inheritdoc/>
        public override Expr Visit(Tuple expr)
        {

            return expr with
            {
                Fields = new(expr.Fields.Select(Visit)),
            };
        }

        /// <inheritdoc/>
        public override Expr Visit(Var expr)
        {
            return expr;
        }

        /// <inheritdoc/>
        public override Expr Visit(TIR.IterVar expr)
        {
            return expr;
        }

        /// <inheritdoc/>
        public override Expr Visit(TIR.Sequential expr)
        {
            return expr with
            {
                Fields = new(expr.Fields.Select(Visit)),
            };
        }

        /// <inheritdoc/>
        public override Expr Visit(TIR.For expr)
        {
            return expr with
            {
                LoopVar = (Var)Visit(expr.LoopVar),
                Dom = VisitRange(expr.Dom),
                Body = (TIR.Sequential)Visit(expr.Body),
            };
        }

        public virtual TIR.Range VisitRange(TIR.Range range)
        {
            return range with
            {
                Min = Visit(range.Min),
                Max = Visit(range.Max)
            };
        }

        public virtual TIR.BufferRegion VisitBufferRegion(TIR.BufferRegion region)
        {
            return region;
        }

        public virtual TIR.Buffer VisitBuffer(TIR.Buffer buffer)
        {
            return buffer with
            {
                Handle = (Var)Visit(buffer.Handle),
                Shape = (IR.Tuple)Visit(buffer.Shape),
                Strides = (IR.Tuple)Visit(buffer.Strides),
                ElemOffset = Visit(buffer.ElemOffset)
            };
        }

        public override Expr Visit(TIR.Block expr)
        {
            return expr with
            {
                // the block realize 
                InitBody = (TIR.Sequential)Visit(expr.InitBody),
                Predicate = Visit(expr.Predicate),
                IterVars = VisitArrayList(expr.IterVars, x => (TIR.IterVar)Visit(x)),
                // the block internal.
                Body = (TIR.Sequential)Visit(expr.Body),
                Reads = new(expr.Reads.Select(VisitBufferRegion)),
                Writes = new(expr.Writes.Select(VisitBufferRegion)),
                AllocBuffers = new(expr.AllocBuffers.Select(VisitBuffer)),
            };
        }

        /// <inheritdoc/>
        public override Expr Visit(TIR.BufferStore expr)
        {
            return expr with
            {
                Indices = VisitArray(expr.Indices, Visit),
                Value = Visit(expr.Value)
            };
        }

        /// <inheritdoc/>
        public override Expr Visit(TIR.BufferLoad expr)
        {
            return expr with
            {
                Indices = VisitArray(expr.Indices, Visit),
            };
        }

        public virtual IRArrayList<TResult> VisitArrayList<TInput, TResult>(IRArrayList<TInput> arrayList, Func<TInput, TResult> visitor)
        {
            return new(arrayList.Select(visitor));
        }

        public virtual IRArray<TResult> VisitArray<TInput, TResult>(IRArray<TInput> array, Func<TInput, TResult> visitor)
        {
            return new(array.Select(visitor));
        }

    }
}
