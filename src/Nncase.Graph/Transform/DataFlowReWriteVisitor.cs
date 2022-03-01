// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Nncase.IR;
using Nncase.PatternMatch;

namespace Nncase.Transform;

/// <summary>
/// DataFlowReWriteVisitor.
/// </summary>
internal sealed class DataFlowRewriteVisitor : ExprMutator
{
    private readonly IRewriteRule _rule;

    public DataFlowRewriteVisitor(IRewriteRule rule)
    {
        _rule = rule;
    }

    public override Expr DefaultMutateLeaf(Expr expr)
    {
        if (CompilerServices.TryMatchRoot(expr, _rule.Pattern, out var match))
        {
            var replace = _rule.GetReplace(match);
            if (replace != null)
            {
                return replace;
            }
        }

        return expr;
    }
}
