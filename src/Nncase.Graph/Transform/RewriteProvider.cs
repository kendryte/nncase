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

namespace Nncase.Transform;

internal class RewriteProvider : IRewriteProvider
{
    public Expr Rewrite(Expr expr, IEnumerable<IRewriteRule> rules, RunPassContext context)
    {
        if (expr.CheckedType == null)
        {
            CompilerServices.InferenceType(expr);
        }

        var post = expr;
        int count = 0;
        OnRewriteStart(expr, context, count);
        do
        {
            bool isMutated = false;
            foreach (var rule in rules)
            {
                var visitor = new DataFlowRewriteVisitor(rule, context);
                var last = post;
                post = visitor.Visit(last);
                if (visitor.IsMutated)
                {
                    isMutated = true;
                    break;
                }
            }

            var inferSuccess = CompilerServices.InferenceType(post);
            OnRewriteEnd(post, context, count++);
            if (!inferSuccess && DumpScope.Current.IsEnabled(DumpFlags.Rewrite))
            {
                DumpScope.Current.DumpIR(expr, $"{count}_End_InferFailed", "Rewrite");
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
