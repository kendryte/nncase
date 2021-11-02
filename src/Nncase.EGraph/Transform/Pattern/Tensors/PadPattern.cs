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
    public sealed record PadPattern(Func<Pad, bool> Cond) : OpPattern
    {
        public PadPattern(Pad pad) : this(x => x == pad) { }

        public PadPattern(PadMode padMode) : this(x => x.padMode == padMode) { }

        public bool MatchLeaf(Pad pad)
        {
            return Cond(pad) && MatchCheckedType(pad);
        }
    }
}
