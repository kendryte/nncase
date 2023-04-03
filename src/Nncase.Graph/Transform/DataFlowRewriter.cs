// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Reactive;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.PatternMatch;
using Nncase.Utilities;

namespace Nncase.Passes;

/// <summary>
/// Dataflow rewriter.
/// </summary>
internal sealed class DataFlowRewriter : ExprRewriter
{
    private readonly IRewriteRule _rule;
    private readonly RunPassContext _options;
    private readonly HashSet<Expr> _dontInheritExprs = new HashSet<Expr>(ReferenceEqualityComparer.Instance);

    public DataFlowRewriter(IRewriteRule rule, RunPassContext options)
    {
        _rule = rule;
        _options = options;
    }

    protected override Expr DefaultRewriteLeaf(Expr expr)
    {
        if (CompilerServices.TryMatchRoot(expr, _rule.Pattern, _options.MatchOptions, out var match))
        {
            var replace = _rule.GetReplace(match, _options)?.InheritMetaData(expr);
            if (replace != null)
            {
                _dontInheritExprs.Add(replace);

                return replace;
            }
        }

        return expr;
    }

    protected override Expr DispatchVisit(Expr expr, Unit context)
    {
        var replace = base.DispatchVisit(expr, context);
        if (!_dontInheritExprs.Contains(expr))
        {
            _options.MatchOptions.InheritSuppressPatterns(expr, replace);
        }

        return replace;
    }
}
