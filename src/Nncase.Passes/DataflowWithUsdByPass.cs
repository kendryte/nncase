// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Reactive;
using System.Threading.Tasks;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.PatternMatch;

namespace Nncase.Passes;

/// <summary>
/// the rule for <see cref="DataflowWithUsdByPass"/>.
/// </summary>
public interface IRewriteRuleWithUsdBy
{
    /// <summary>
    /// Sets set the UsedByResult info.
    /// </summary>
    public IUsedByResult UsedByResult { set; }
}

/// <summary>
/// Dataflow pass.
/// </summary>
public sealed class DataflowWithUsdByPass : DataflowPass
{
    /// <inheritdoc/>
    protected override Task<BaseFunction> RunCoreAsync(BaseFunction function, RunPassContext options)
    {
        return Task.FromResult((BaseFunction)Rewrite(function, Rules, options));
    }

    private Expr Rewrite(Expr expr, IEnumerable<IRewriteRule> rules, RunPassContext options)
    {
        var post = expr;
        int count = 0;
        OnRewriteStart(expr, options, count);
        do
        {
            bool isMutated = false;
            foreach (var rule in rules)
            {
                var visitor = new DataflowWithUsdByVisitor(rule, options);
                var last = post;
                if (rule is IRewriteRuleWithUsdBy usedbyRule)
                {
                    usedbyRule.UsedByResult = Analyser.AnalysisUsedBy(last);
                }

                post = visitor.Rewrite(last);
                if (visitor.IsMutated)
                {
                    isMutated = true;
                    break;
                }
            }

            var inferSuccess = CompilerServices.InferenceType(post);
            OnRewriteEnd(post, options, count++);
            if (isMutated && !inferSuccess)
            {
                if (DumpScope.Current.IsEnabled(DumpFlags.Rewrite))
                {
                    DumpScope.Current.DumpIR(post, $"InferShape_{count - 1}_Failed", "RewriteFailed");
                }

                throw new InvalidOperationException($"After Rewrite {count - 1}, InferShape Failed For This Model!");
            }

            if (!isMutated || options.RewriteOnce)
            {
                break;
            }
        }
        while (true);
        return post;
    }

    /// <summary>
    /// callback for rewrite start.
    /// </summary>
    private void OnRewriteStart(Expr expr, RunPassContext options, int count)
    {
        if (DumpScope.Current.IsEnabled(DumpFlags.Rewrite))
        {
            DumpScope.Current.DumpIR(expr, $"{count}_Start", "Rewrite");
        }
    }

    /// <summary>
    /// call back for rewrite end.
    /// </summary>
    private void OnRewriteEnd(Expr expr, RunPassContext options, int count)
    {
        if (DumpScope.Current.IsEnabled(DumpFlags.Rewrite))
        {
            DumpScope.Current.DumpIR(expr, $"{count}_End", "Rewrite");
        }
    }
}

/// <summary>
/// DataflowWithUsdByVisitor.
/// </summary>
internal sealed class DataflowWithUsdByVisitor : ExprRewriter
{
    private readonly IRewriteRule _rule;
    private readonly RunPassContext _options;
    private readonly HashSet<Expr> _dontInheritExprs = new HashSet<Expr>(ReferenceEqualityComparer.Instance);

    public DataflowWithUsdByVisitor(IRewriteRule rule, RunPassContext options)
    {
        _rule = rule;
        _options = options;
    }

    protected override Expr DefaultRewriteLeaf(Expr expr)
    {
        if (!IsMutated && CompilerServices.TryMatchRoot(expr, _rule.Pattern, _options.MatchOptions, out var match))
        {
            var replace = _rule.GetReplace(match, _options);
            if (replace != null)
            {
                replace.CheckedType = expr.CheckedType;
                _dontInheritExprs.Add(replace);
                return replace;
            }
        }

        return expr;
    }

    protected override Expr DispatchVisit(Expr expr, Unit context)
    {
        var newExpr = base.DispatchVisit(expr, context);
        if (!_dontInheritExprs.Contains(expr))
        {
            _options.MatchOptions.InheritSuppressPatterns(expr, newExpr);
        }

        return newExpr;
    }
}
