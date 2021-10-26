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
    /// Clamp expression.
    /// </summary>
    public record ClampPattern(Func<Clamp, bool> Cond) : OpPattern
    {
        public ClampPattern(Clamp clamp) : this(x => x == clamp) { }

        public bool MatchLeaf(Clamp clamp)
        {
            return Cond(clamp) && MatchCheckedType(clamp);
        }
    }
}
