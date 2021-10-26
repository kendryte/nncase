// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR.Tensors;
namespace Nncase.Transform.Pattern.Tensors
{
    /// <summary>
    /// Transpose expression.
    /// </summary>
    public record TransposePattern(Func<Transpose, bool> Cond) : OpPattern
    {
        public TransposePattern(Transpose transpose) : this(x => x == transpose) { }

        public bool MatchLeaf(Transpose transpose)
        {
            return Cond(transpose) && MatchCheckedType(transpose);
        }
    }
}
