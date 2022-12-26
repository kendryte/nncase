// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.Transform;

internal class EGraphRewriteProvider : IEGraphRewriteProvider
{
    public Expr ERewrite(Expr expr, IEnumerable<IRewriteRule> rules, RunPassOptions options)
    {
        if (expr.CheckedType is null)
        {
            CompilerServices.InferenceType(expr);
        }

        var graph = new EGraph();
        var root = graph.Add(expr);
        EGraphRewriter.Rewrite(graph, rules, options);
        var post = graph.Extract(root, null, options);
        return post;
    }
}
