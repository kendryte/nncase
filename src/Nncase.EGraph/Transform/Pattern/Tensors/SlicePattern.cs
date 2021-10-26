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
    public record SlicePattern(Func<Slice, bool> Cond) : OpPattern
    {
        public SlicePattern(Slice slice) : this(x => x == slice) { }

        public bool MatchLeaf(Slice slice)
        {
            return Cond(slice) && MatchCheckedType(slice);
        }
    }
}
