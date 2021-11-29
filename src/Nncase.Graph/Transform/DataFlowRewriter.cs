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

    public static class DataFlowRewrite
    {
        private static Expr RewriteImpl(Expr pre, IEnumerable<PatternRule> Rules, RunPassOptions options)
        {
            var visitor = new DataFlowReWriteVisitor();
            var post = pre;
            var last = post;
            int count = 0;
            var dumpPath = Path.Combine(options.FullDumpDir, "Rewrite");
            do
            {
                if (options.DumpLevel > 2)
                    IRPrinter.DumpExprAsIL(pre, $"{count}_Before", dumpPath);
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
                            TypeInference.InferenceType(post);
                            break;
                        }
                    }
                    if (visitor.isMatched)
                        break;
                }
                if (options.DumpLevel > 2)
                    IRPrinter.DumpExprAsIL(post, $"{count++}_After", dumpPath);
                if (!visitor.isMatched)
                    break;
            } while (true);
            return post;
        }

        public static Expr Rewrite(Expr pre, IEnumerable<PatternRule> Rules, RunPassOptions options) => RewriteImpl(pre, Rules, options);
    }

}