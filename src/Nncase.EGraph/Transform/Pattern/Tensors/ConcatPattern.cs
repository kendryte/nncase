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
    /// Concat expression.
    /// </summary>
    public record ConcatPattern(Func<Concat, bool> Cond) : OpPattern
    {
        public ConcatPattern(Concat concat) : this(x => x == concat) { }

        public bool MatchLeaf(Concat concat)
        {
            return Cond(concat) && MatchCheckedType(concat);
        }
    }
}
