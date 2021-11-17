// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Nncase.IR;

namespace Nncase.Pattern
{
    public sealed record IfPattern(ExprPattern Cond, ExprPattern TrueBranch, ExprPattern FalseBranch) : ExprPattern
    {
        public IfPattern(If expr) : this((ExprPattern)expr.Cond, (ExprPattern)expr.TrueBranch, (ExprPattern)expr.FalseBranch)
        { }

        public bool MatchLeaf(If expr)
        {
            return MatchCheckedType(expr);
        }

    }
    public static partial class Utility
    {
        public static IfPattern IsIf(ExprPattern Cond, ExprPattern TrueBranch, ExprPattern FalseBranch) => new IfPattern(Cond, TrueBranch, FalseBranch);
    }
}