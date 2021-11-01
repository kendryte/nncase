// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.IR.Tensors;

namespace Nncase.Transform.Pattern.Tensors
{
    public record ReducePattern(Func<Reduce, bool> Cond) : OpPattern
    {
        public ReducePattern(Reduce reduce) : this(x => x == reduce) { }
        public ReducePattern(ReduceOp reduceOp) : this(x => x.reduceOp == reduceOp) { }

        public bool MatchLeaf(Reduce reduce)
        {
            return Cond(reduce) && MatchCheckedType(reduce);
        }
    }
}
