// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.IO;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.Pattern;

namespace Nncase.Transform
{
    /// <summary>
    /// rewrite method.
    /// </summary>
    public static class DataFlowRewrite
    {
        /// <summary>
        /// callback for rewrite start.
        /// </summary>
        /// <param name="expr"></param>
        /// <param name="options"></param>
        /// <param name="count"></param>
        private static void OnRewriteStart(Expr expr, RunPassOptions options, int count)
        {
            switch (options.DumpLevel)
            {
                case >= 2:
                    IRPrinter.DumpExprAsIL(expr, $"{count}_Start", Path.Combine(options.FullDumpDir, "Rewrite"));
                    break;
                case >= 1:
                    expr.DumpExprAsIL();
                    break;
                default:
                    break;
            }
        }

        /// <summary>
        /// call back for rewrite end.
        /// </summary>
        /// <param name="expr"></param>
        /// <param name="options"></param>
        /// <param name="count"></param>
        private static void OnRewriteEnd(Expr expr, RunPassOptions options, int count)
        {
            switch (options.DumpLevel)
            {
                case >= 2:
                    IRPrinter.DumpExprAsIL(expr, $"{count}_End", Path.Combine(options.FullDumpDir, "Rewrite"));
                    break;
                case >= 1:
                    expr.DumpExprAsIL();
                    break;
                default:
                    break;
            }
        }

        private static Expr RewriteImpl(Expr pre, IEnumerable<PatternRule> Rules, RunPassOptions options)
        {
            var visitor = new DataFlowReWriteVisitor();
            var post = pre;
            var last = post;
            int count = 0;
            do
            {
                OnRewriteStart(last, options, count);
                foreach (var rule in Rules)
                {
                    visitor.Rule = rule;
                    foreach (var pattern in rule.Patterns)
                    {
                        visitor.Pattern = pattern;
                        last = post;
                        visitor.Clear();
                        post = visitor.Visit(last);
                        if (visitor.isMatched)
                        {
                            CompilerServices.InferenceType(post);
                            break;
                        }
                    }

                    if (visitor.isMatched)
                        break;
                }

                OnRewriteEnd(post, options, count++);
                if (!visitor.isMatched)
                    break;
            } while (true);
            return post;
        }

        public static Expr Rewrite(Expr pre, IEnumerable<PatternRule> Rules, RunPassOptions options) => RewriteImpl(pre, Rules, options);
    }
}