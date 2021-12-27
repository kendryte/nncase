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
    public abstract class ExprMatutor : ExprFunctor<Expr, IRType>
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
                Body = (TIR.Sequential)Visit(expr.Body),
                InitBody = (TIR.Sequential)Visit(expr.InitBody),
                IterVarPairs = new(expr.IterVarPairs.Select(t => (
                 (TIR.IterVar)Visit(t.iterVar),
                 (Var)Visit(t.loopVar)
                ))),
                Reads = new(expr.Reads.Select(VisitBufferRegion)),
                Writes = new(expr.Writes.Select(VisitBufferRegion)),
                AllocBuffers = new(expr.AllocBuffers.Select(VisitBuffer)),
                Predicate = Visit(expr.Predicate)
            };
        }
    }
}
