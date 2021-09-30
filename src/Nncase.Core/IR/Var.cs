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
    /// Variable expression.
    /// </summary>
    public sealed record Var(string Name, IRType? TypeAnnotation = null) : Expr
    {
        private static int _globalVarIndex = 0;

        /// <summary>
        /// Initializes a new instance of the <see cref="Var"/> class.
        /// </summary>
        /// <param name="typeAnnotation">Type annotation.</param>
        public Var(IRType? typeAnnotation)
            : this($"var_{_globalVarIndex++}", typeAnnotation)
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="Var"/> class.
        /// </summary>
        public Var()
            : this($"var_{_globalVarIndex++}", null)
        {
        }
    }
}
