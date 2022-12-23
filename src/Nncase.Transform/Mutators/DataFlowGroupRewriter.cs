// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.PatternMatch;

namespace Nncase.Transform;

/// <summary>
/// rewrite method.
/// </summary>
public sealed class DataFlowMergeRewriter
{
    /// <summary>
    /// Rewrite the merge rule.
    /// </summary>
    /// <param name="expr"></param>
    /// <param name="rules"></param>
    /// <param name="mutator_creator"></param>
    /// <param name="options"></param>
    /// <returns></returns>
    /// <exception cref="InvalidOperationException"></exception>
    public Expr Rewrite(Expr expr, IEnumerable<Mutators.IMergeRewriteRule> rules, Func<IUsedByResult, Mutators.IMergeRewriteRule, RunPassOptions, Mutators.FusionGroupMutator> mutator_creator, RunPassOptions options)
    {
        var post = expr;
        int count = 0;
        OnRewriteStart(expr, options, count);
        do
        {
            bool isMutated = false;
            foreach (var rule in rules)
            {
                var last = post;

                // todo reduce the mutator and rules dependence.
                var visitor = mutator_creator(Analyser.AnalysisUsedBy(last), rule, options);
                post = visitor.Visit(last);
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
                if (options.DumpLevel > 1)
                {
                    CompilerServices.DumpIR(post, $"InferShape_{count - 1}_Failed", options.DumpDir);
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
    private void OnRewriteStart(Expr expr, RunPassOptions options, int count)
    {
        switch (options.DumpLevel)
        {
            case >= 2:
                CompilerServices.DumpIR(expr, $"{count}_Start", Path.Combine(options.DumpDir, "Rewrite"));
                break;
            case >= 1:
                break;
            default:
                break;
        }
    }

    /// <summary>
    /// call back for rewrite end.
    /// </summary>
    private void OnRewriteEnd(Expr expr, RunPassOptions options, int count)
    {
        switch (options.DumpLevel)
        {
            case >= 2:
                CompilerServices.DumpIR(expr, $"{count}_End", Path.Combine(options.DumpDir, "Rewrite"));
                break;
            case >= 1:
                break;
            default:
                break;
        }
    }
}
