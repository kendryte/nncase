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
    /// Type functor.
    /// </summary>
    /// <typeparam name="TResult">Result visit type.</typeparam>
    public abstract class TypeFunctor<TResult>
    {
        /// <summary>
        /// Visit type.
        /// </summary>
        /// <param name="type">Type.</param>
        /// <returns>Result.</returns>
        public virtual TResult VisitType(IRType type)
        {
            return type switch
            {
                AnyType t => VisitType(t),
                InvalidType t => VisitType(t),
                TensorType t => VisitType(t),
                TupleType t => VisitType(t),
                CallableType t => VisitType(t),
                PointerType t => VisitType(t),
                _ => DefaultVisitType(type),
            };
        }

        /// <summary>
        /// Visit any type.
        /// </summary>
        /// <param name="type">Any type.</param>
        /// <returns>Result.</returns>
        public virtual TResult VisitType(AnyType type) => DefaultVisitType(type);

        /// <summary>
        /// Visit invalid type.
        /// </summary>
        /// <param name="type">Invalid type.</param>
        /// <returns>Result.</returns>
        public virtual TResult VisitType(InvalidType type) => DefaultVisitType(type);

        /// <summary>
        /// Visit tensor type.
        /// </summary>
        /// <param name="type">Tensor type.</param>
        /// <returns>Result.</returns>
        public virtual TResult VisitType(TensorType type) => DefaultVisitType(type);

        /// <summary>
        /// Visit tuple type.
        /// </summary>
        /// <param name="type">Tuple type.</param>
        /// <returns>Result.</returns>
        public virtual TResult VisitType(TupleType type) => DefaultVisitType(type);

        /// <summary>
        /// Visit callable type.
        /// </summary>
        /// <param name="type">Callable type.</param>
        /// <returns>Result.</returns>
        public virtual TResult VisitType(CallableType type) => DefaultVisitType(type);

        /// <summary>
        /// Visit pointer type expression.
        /// </summary>
        /// <param name="type">Callable type.</param>
        /// <returns>Result.</returns>
        public virtual TResult VisitType(PointerType type) => DefaultVisitType(type);

        /// <summary>
        /// Default visit routine.
        /// </summary>
        /// <param name="type">Type.</param>
        /// <returns>Result.</returns>
        public virtual TResult DefaultVisitType(IRType type)
        {
            throw new NotImplementedException($"Unhandled visit routine for {type.GetType()}.");
        }
    }
}
