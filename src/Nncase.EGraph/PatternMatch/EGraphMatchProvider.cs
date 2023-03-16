// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.Passes;

namespace Nncase.PatternMatch;

internal class EGraphMatchProvider : IEGraphMatchProvider
{
    public bool TryMatchRoot(IEnumerable<ENode> enodes, IPattern pattern, [MaybeNullWhen(false)] out IReadOnlyList<IMatchResult> results)
    {
        return EGraphMatcher.TryMatchRoot(enodes, pattern, out results);
    }

    public bool TryEMatchRoot(Expr expr, IPattern pattern, [MaybeNullWhen(false)] out IReadOnlyList<IMatchResult> results)
    {
        var egraph = new EGraph(expr);
        return TryMatchRoot(egraph.Nodes, pattern, out results);
    }
}
