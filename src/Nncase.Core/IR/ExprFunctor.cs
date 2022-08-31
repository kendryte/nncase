// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.IR
{
    /// <summary>
    /// Expression functor.
    /// </summary>
    /// <typeparam name="TExprResult">Expression visit result type.</typeparam>
    /// <typeparam name="TTypeResult">Type visit result type.</typeparam>
    public abstract class ExprFunctor<TExprResult, TTypeResult> : TypeFunctor<TTypeResult>
    {
        /// <summary>
        /// Visit expression.
        /// </summary>
        /// <param name="expr">Expression.</param>
        /// <returns>Result.</returns>
        public virtual TExprResult Visit(Expr expr)
        {
            return expr switch
            {
                Var var => Visit(var),
                Const con => Visit(con),
                Function func => Visit(func),
                Fusion fusion => Visit(fusion),
                Call call => Visit(call),
                Tuple tuple => Visit(tuple),
                Op op => Visit(op),
                None none => Visit(none),
                Marker marker => Visit(marker),
                PrimFunctionWrapper wrapper => Visit(wrapper),
                TIR.IterVar itvar => Visit(itvar),
                TIR.Sequential seq => Visit(seq),
                TIR.For @for => Visit(@for),
                TIR.Block block => Visit(block),
                TIR.BufferLoad bload => Visit(bload),
                TIR.BufferStore bstore => Visit(bstore),
                TIR.IfThenElse ift => Visit(ift),
                TIR.PrimFunction primfunc => Visit(primfunc),
                TIR.Let let => Visit(let),
                TIR.Buffer buffer => Visit(buffer),
                TIR.BufferRegion region => Visit(region),
                _ => DefaultVisit(expr),
            };
        }

        /// <summary>
        /// Visit variable expression.
        /// </summary>
        /// <param name="expr">Variable expression.</param>
        /// <returns>Result.</returns>
        public virtual TExprResult Visit(Var expr) => DefaultVisit(expr);

        /// <summary>
        /// Visit constant expression.
        /// </summary>
        /// <param name="expr">Constant expression.</param>
        /// <returns>Result.</returns>
        public virtual TExprResult Visit(Const expr) => DefaultVisit(expr);

        /// <summary>
        /// Visit function expression.
        /// </summary>
        /// <param name="expr">Variable expression.</param>
        /// <returns>Result.</returns>
        public virtual TExprResult Visit(Function expr) => DefaultVisit(expr);

        /// <summary>
        /// Visit fusion expression
        /// </summary>
        /// <param name="expr">Fusion Expression</param>
        /// <returns></returns>
        public virtual TExprResult Visit(Fusion expr) => DefaultVisit(expr);

        /// <summary>
        /// Visit prim function wrapper expression.
        /// </summary>
        /// <param name="expr">PrimFunctionWrapper expression.</param>
        /// <returns>Result.</returns>
        public virtual TExprResult Visit(PrimFunctionWrapper expr) => DefaultVisit(expr);

        /// <summary>
        /// Visit prim function expression.
        /// </summary>
        /// <param name="expr">Variable expression.</param>
        /// <returns>Result.</returns>
        public virtual TExprResult Visit(TIR.PrimFunction expr) => DefaultVisit(expr);

        /// <summary>
        /// Visit call expression.
        /// </summary>
        /// <param name="expr">Call expression.</param>
        /// <returns>Result.</returns>
        public virtual TExprResult Visit(Call expr) => DefaultVisit(expr);

        /// <summary>
        /// Visit tuple expression.
        /// </summary>
        /// <param name="expr">Variable expression.</param>
        /// <returns>Result.</returns>
        public virtual TExprResult Visit(Tuple expr) => DefaultVisit(expr);

        /// <summary>
        /// Visit operator expression.
        /// </summary>
        /// <param name="expr">Operator expression.</param>
        /// <returns>Result.</returns>
        public virtual TExprResult Visit(Op expr) => DefaultVisit(expr);

        /// <summary>
        /// Visit None expression
        /// </summary>
        /// <param name="expr">None expr.</param>
        /// <returns>Result.</returns>
        public virtual TExprResult Visit(None expr) => DefaultVisit(expr);

        /// <summary>
        /// Visit marker expression
        /// </summary>
        /// <param name="expr">Marker expr.</param>
        /// <returns>Result.</returns>
        public virtual TExprResult Visit(Marker expr) => DefaultVisit(expr);

        /// <summary>
        /// Visit IterVar expression.
        /// </summary>
        /// <param name="expr">IterVar expression.</param>
        /// <returns>Result.</returns>
        public virtual TExprResult Visit(TIR.IterVar expr) => DefaultVisit(expr);

        /// <summary>
        /// Visit Sequential expression.
        /// </summary>
        /// <param name="expr">Sequential expression.</param>
        /// <returns>Result.</returns>
        public virtual TExprResult Visit(TIR.Sequential expr) => DefaultVisit(expr);

        /// <summary>
        /// Visit For expression.
        /// </summary>
        /// <param name="expr">For expression.</param>
        /// <returns>Result.</returns>
        public virtual TExprResult Visit(TIR.For expr) => DefaultVisit(expr);

        /// <summary>
        /// Visit block expression.
        /// </summary>
        /// <param name="expr">block expression.</param>
        /// <returns>Result.</returns>
        public virtual TExprResult Visit(TIR.Block expr) => DefaultVisit(expr);

        /// <summary>
        /// Visit BufferLoad expression.
        /// </summary>
        /// <param name="expr">BufferLoad expression.</param>
        /// <returns>Result.</returns>
        public virtual TExprResult Visit(TIR.BufferLoad expr) => DefaultVisit(expr);

        /// <summary>
        /// Visit BufferStore expression.
        /// </summary>
        /// <param name="expr">BufferStore expression.</param>
        /// <returns>Result.</returns>
        public virtual TExprResult Visit(TIR.BufferStore expr) => DefaultVisit(expr);

        /// <summary>
        /// Visit IfThenElse expression.
        /// </summary>
        /// <param name="expr">IfThenElse expression.</param>
        /// <returns>Result.</returns>
        public virtual TExprResult Visit(TIR.IfThenElse expr) => DefaultVisit(expr);

        /// <summary>
        /// Visit Let expression.
        /// </summary>
        /// <param name="expr">Let expression.</param>
        /// <returns>Result.</returns>
        public virtual TExprResult Visit(TIR.Let expr) => DefaultVisit(expr);

        /// <summary>
        /// Visit MemRef expression.
        /// </summary>
        /// <param name="expr">MemRef expression.</param>
        /// <returns>Result.</returns>
        public virtual TExprResult Visit(TIR.PhysicalBuffer expr) => DefaultVisit(expr);

        /// <summary>
        /// Visit buffer region expression.
        /// </summary>
        /// <param name="expr">buffer region expression.</param>
        /// <returns>Result.</returns>
        public virtual TExprResult Visit(TIR.BufferRegion expr) => DefaultVisit(expr);

        /// <summary>
        /// Default visit routine.
        /// </summary>
        /// <param name="expr">Expression.</param>
        /// <returns>Result.</returns>
        public virtual TExprResult DefaultVisit(Expr expr)
        {
            throw new NotImplementedException($"Unhandled visit routine for {expr.GetType()}.");
        }
    }
}
