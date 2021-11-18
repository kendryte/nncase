// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
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
        public static Expr RewriteDataFlow(Expr pre, List<PatternRule> Rules)
        {
            var visitor = new DataFlowReWriteVisitor();
            var post = pre;
            var last = post;
            do
            {
                visitor.Clear();
                foreach (var rule in Rules)
                {
                    visitor.Rule = rule;
                    foreach (var pattern in rule.Patterns)
                    {
                        visitor.Pattern = pattern;
                        last = post;
                        post = visitor.Visit(last);
                        if (visitor.isMatched)
                            break;
                    }
                    if (visitor.isMatched)
                        break;
                }
            } while (!visitor.isMatched);
            return post;
        }
    }
}