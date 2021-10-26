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
    public record BinaryPattern(Func<Binary, bool> Cond) : OpPattern
    {
        public BinaryPattern(Binary binary) : this(x => x == binary) { }
        public BinaryPattern(BinaryOp binaryOp) : this(x => x.BinaryOp == binaryOp) { }

        public bool MatchLeaf(Binary binary)
        {
            return Cond(binary) && MatchCheckedType(binary);
        }
    }

}
