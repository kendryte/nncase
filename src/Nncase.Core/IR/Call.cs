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
    /// Call expression.
    /// </summary>
    public sealed record Call(Expr Target, ImmutableArray<Expr> Parameters) : Expr
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="Call"/> class.
        /// </summary>
        /// <param name="target">Call target.</param>
        /// <param name="parameters">Parameters.</param>
        public Call(Expr target, params Expr[] parameters)
            : this(target, ImmutableArray.Create(parameters))
        {
        }
    }
}
