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
    private readonly RunPassOptions _options;
    private readonly HashSet<Expr> _dontInheritExprs = new HashSet<Expr>(ReferenceEqualityComparer.Instance);

    public DataFlowRewriteVisitor(IRewriteRule rule, RunPassOptions options)
    {
        _rule = rule;
        _options = options;
    }

    /// <summary>
    /// the rule dataflow rewrite can't mutate fusion.
    /// </summary>
    /// <param name="fusion"></param>
    /// <returns></returns>
    public override Expr MutateLeaf(Fusion fusion)
    {
        return fusion;
    }

    public override Expr DefaultMutateLeaf(Expr expr)
    {
        if (CompilerServices.TryMatchRoot(expr, _rule.Pattern, _options.MatchOptions, out var match))
        {
            var replace = _rule.GetReplace(match, _options);
            if (replace != null)
            {
                _dontInheritExprs.Add(replace);
                return replace;
            }
        }

        return expr;
    }

    public override Expr Visit(Expr expr)
    {
        var newExpr = base.Visit(expr);
        if (!_dontInheritExprs.Contains(expr))
        {
            _options.MatchOptions.InheritSuppressPatterns(expr, newExpr);
        }

        return newExpr;
    }
}
