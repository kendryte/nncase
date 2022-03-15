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
    /// the Callable Expr
    /// </summary>
    public abstract record Callable(string Name) : Expr
    {

        /// <summary>
        /// the schedule result, the dag function for stackvm, the prim_func for other backend.
        /// </summary>
        public Schedule.SchedFunctionResult? SchedResult = null;
    }

    /// <summary>
    /// Function expression.
    /// </summary>
    public sealed record Function(string Name, Expr Body, IRArray<Var> Parameters) : Callable(Name)
    {
        private static int _globalFuncIndex = 0;

        /// <summary>
        /// Initializes a new instance of the <see cref="Function"/> class.
        /// </summary>
        /// <param name="parameters">Parameters.</param>
        /// <param name="body">Body.</param>
        public Function(Expr body, IRArray<Var> parameters)
            : this($"func_{_globalFuncIndex++}", body, parameters)
        {
        }

        /// <summary>
        /// build function.
        /// </summary>
        /// <param name="body"></param>
        /// <param name="parameters"></param>
        public Function(Expr body, params Var[] parameters)
            : this($"func_{_globalFuncIndex++}", body, new(parameters))
        {
        }
    }
}
