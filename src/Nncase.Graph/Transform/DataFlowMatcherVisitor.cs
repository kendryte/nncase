// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.IR;
using Nncase.PatternMatch;
using ContextEnv = System.Collections.Generic.Dictionary<Nncase.PatternMatch.ExprPattern, Nncase.IR.Expr>;
using Tuple = Nncase.IR.Tuple;

namespace Nncase.Transform;

public static class DataFlowMatcher
{
    /// <summary>
    /// Match the Expr with Pattern.
    /// </summary>
    /// <param name="expr"></param>
    /// <param name="pattern"></param>
    /// <returns> bool. </returns>
    public static List<IMatchResult> Match(Expr expr, ExprPattern pattern)
    {
        if (expr.CheckedType is null) { expr.InferenceType(); }
        var results = new List<IMatchResult>();
        var matcher = new DataFlowMatcherVisitor();
        if (matcher.Visit(pattern, expr))
        {
            results.Add(new DFMatchResult(expr, matcher.Env));
        }

        return results;
    }
}
