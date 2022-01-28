// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.IR;
using static Nncase.IR.Utility;

namespace Nncase.Pattern
{
    /// <summary>
    /// The Or Pattern for Match Different branch, NOTE if both branch are matched, choice the Lhs.
    /// </summary>
    /// <param name="Lhs"></param>
    /// <param name="Rhs"></param>
    public sealed record OrPattern(ExprPattern Lhs, ExprPattern Rhs) : ExprPattern
    {
        public override ExprPattern Copy() => this with { Id = _globalPatIndex++, Lhs = Lhs.Copy(), Rhs = Rhs.Copy() };

        public override void Clear()
        {
            Lhs.Clear();
            Rhs.Clear();
        }
    }
}