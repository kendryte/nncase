// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.Diagnostics;
using Nncase.IR;

namespace Nncase.Passes;

internal class RewriteProvider : IRewriteProvider
{
    public Expr Rewrite(Expr expr, IEnumerable<IRewriteRule> rules, RunPassContext context)
    {
        CompilerServices.InferenceType(expr);
        IRewriteRule? lastRule = null;
        var post = expr;
        int count = 0;
        OnRewriteStart(expr, context, count);
        do
        {
            bool isMutated = false;
            bool switchRule = false;
            foreach (var rule in rules)
            {
                var visitor = new DataFlowRewriter(rule, context);
                var last = post;
                post = visitor.Rewrite(last);
                if (visitor.IsMutated)
                {
                    isMutated = true;
                    if (!ReferenceEquals(lastRule, rule))
                    {
                        lastRule = rule;
                        switchRule = true;
                    }

                    break;
                }
            }

            var inferSuccess = CompilerServices.InferenceType(post);
            if (switchRule)
            {
                OnRewriteEnd(post, context, count++);
            }

            if (!inferSuccess && DumpScope.Current.IsEnabled(DumpFlags.Rewrite))
            {
                DumpScope.Current.DumpIR(post, $"{count}_End_InferFailed", "Rewrite");
            }

            Trace.Assert(inferSuccess);

            if (!isMutated || context.RewriteOnce)
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
    private void OnRewriteStart(Expr expr, RunPassContext context, int count)
    {
        if (DumpScope.Current.IsEnabled(DumpFlags.Rewrite))
        {
            DumpScope.Current.DumpIR(expr, $"{count}_Start", "Rewrite");
        }
    }

    /// <summary>
    /// call back for rewrite end.
    /// </summary>
    private void OnRewriteEnd(Expr expr, RunPassContext context, int count)
    {
        if (DumpScope.Current.IsEnabled(DumpFlags.Rewrite))
        {
            DumpScope.Current.DumpIR(expr, $"{count}_End", "Rewrite");
        }
    }
}
