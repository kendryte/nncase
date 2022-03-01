// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.PatternMatch;

internal class MatchProvider : IMatchProvider
{
    public bool TryMatch(Expr expr, IPattern pattern, [MaybeNullWhen(false)] out IMatchResult result)
    {
        return Matcher.TryMatch(expr, pattern, out result);
    }

    public bool TryMatchRoot(Expr expr, IPattern pattern, [MaybeNullWhen(false)] out IMatchResult result)
    {
        return Matcher.TryMatchRoot(expr, pattern, out result);
    }
}
