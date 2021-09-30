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
                Call call => Visit(call),
                Tuple tuple => Visit(tuple),
                Op op => Visit(op),
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
