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
    /// Function expression.
    /// </summary>
    public sealed record Function(string Name, Expr Body, IRArray<Expr> Parameters) : Expr
    {
        private static int _globalFuncIndex = 0;

        /// <summary>
        /// Initializes a new instance of the <see cref="Function"/> class.
        /// </summary>
        /// <param name="parameters">Parameters.</param>
        /// <param name="body">Body.</param>
        public Function(Expr body, IRArray<Expr> parameters)
            : this($"func_{_globalFuncIndex++}", body, parameters)
        {
        }

        public Function(Expr body, params Expr[] parameters)
            : this($"func_{_globalFuncIndex++}", body, new(parameters))
        {
        }
    }
}
