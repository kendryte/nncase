// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.IR
{
    /// <summary>
    /// Type Pattern functor.
    /// </summary>
    /// <typeparam name="TResult">Result visit type.</typeparam>
    public abstract class TypePatternFunctor<TResult>
    {
        /// <summary>
        /// Visit type.
        /// </summary>
        /// <param name="pattern">Type.</param>
        /// <returns>Result.</returns>
        public virtual TResult VisitType(TypePattern pattern)
        {
            return DefaultVisitType(pattern);
        }

        /// <summary>
        /// Default visit routine.
        /// </summary>
        /// <param name="pattern">Type.</param>
        /// <returns>Result.</returns>
        public virtual TResult DefaultVisitType(TypePattern pattern)
        {
            throw new NotImplementedException($"Unhandled visit routine for {pattern.GetType()}.");
        }
    }
}
