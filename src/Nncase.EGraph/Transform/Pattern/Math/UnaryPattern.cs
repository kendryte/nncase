// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR.Math;

namespace Nncase.Transform.Pattern.Math
{
    /// <summary>
    /// Unary expression.
    /// </summary>
    public record UnaryPattern(Func<Unary, bool> Cond) : OpPattern
    {

        public UnaryPattern(Unary unary) : this(x => x == unary) { }

        public UnaryPattern(UnaryOp unaryOp) : this(x => x.UnaryOp == unaryOp) { }

        public bool MatchLeaf(Unary unary)
        {
            return Cond(unary) && MatchCheckedType(unary);
        }

    }
}
