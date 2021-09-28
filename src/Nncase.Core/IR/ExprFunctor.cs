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
    /// <typeparam name="TResult">Result type.</typeparam>
    public class ExprFunctor<TResult>
    {
        /// <summary>
        /// Visit expression.
        /// </summary>
        /// <param name="expr">Expression.</param>
        /// <returns>Result.</returns>
        public virtual TResult Visit(Expr expr)
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
        public virtual TResult Visit(Var expr) => DefaultVisit(expr);

        /// <summary>
        /// Visit constant expression.
        /// </summary>
        /// <param name="expr">Constant expression.</param>
        /// <returns>Result.</returns>
        public virtual TResult Visit(Const expr) => DefaultVisit(expr);

        /// <summary>
        /// Visit function expression.
        /// </summary>
        /// <param name="expr">Variable expression.</param>
        /// <returns>Result.</returns>
        public virtual TResult Visit(Function expr) => DefaultVisit(expr);

        /// <summary>
        /// Visit call expression.
        /// </summary>
        /// <param name="expr">Call expression.</param>
        /// <returns>Result.</returns>
        public virtual TResult Visit(Call expr) => DefaultVisit(expr);

        /// <summary>
        /// Visit tuple expression.
        /// </summary>
        /// <param name="expr">Variable expression.</param>
        /// <returns>Result.</returns>
        public virtual TResult Visit(Tuple expr) => DefaultVisit(expr);

        /// <summary>
        /// Visit operator expression.
        /// </summary>
        /// <param name="expr">Operator expression.</param>
        /// <returns>Result.</returns>
        public virtual TResult Visit(Op expr) => DefaultVisit(expr);

        /// <summary>
        /// Default visit routine.
        /// </summary>
        /// <param name="expr">Expression.</param>
        /// <returns>Result.</returns>
        public virtual TResult DefaultVisit(Expr expr)
        {
            throw new NotImplementedException($"Unhandled visit routine for {expr.GetType()}.");
        }
    }
}
