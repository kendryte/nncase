// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.PatternMatch;

/// <summary>
/// Match options.
/// </summary>
public class MatchOptions
{
    /// <summary>
    /// Initializes a new instance of the <see cref="MatchOptions"/> class.
    /// </summary>
    public MatchOptions()
    {
        SuppressedPatterns = new Dictionary<Expr, HashSet<IPattern>>(ReferenceEqualityComparer.Instance);
        RewriteMemo = new Dictionary<Expr, Expr>(ReferenceEqualityComparer.Instance);
    }

    /// <summary>
    /// Gets suppressed patterns.
    /// </summary>
    public Dictionary<Expr, HashSet<IPattern>> SuppressedPatterns { get; }

    /// <summary>
    /// Gets rewrite memo.
    /// </summary>
    public Dictionary<Expr, Expr> RewriteMemo { get; }

    /// <summary>
    /// check the expr and pattern in the suppressed pattern dict.
    /// </summary>
    /// <param name="expr"></param>
    /// <param name="pattern"></param>
    /// <returns></returns>
    public bool IsSuppressedPattern(Expr expr, IPattern pattern)
    {
        if (SuppressedPatterns.TryGetValue(expr, out var patterns))
        {
            return patterns.Contains(pattern);
        }

        return false;
    }

    /// <summary>
    /// add the expr and pattern into suppressed pattern dict.
    /// </summary>
    /// <param name="expr"></param>
    /// <param name="pattern"></param>
    public void SuppressPattern(Expr expr, IPattern pattern)
    {
        if (!SuppressedPatterns.TryGetValue(expr, out var patterns))
        {
            patterns = new HashSet<IPattern>();
            SuppressedPatterns.Add(expr, patterns);
        }

        patterns.Add(pattern);
    }

    /// <summary>
    /// when the soure expr has been changed, need inherit is suppress attribute into new expr.
    /// </summary>
    /// <param name="source"></param>
    /// <param name="dest"></param>
    public void InheritSuppressPatterns(Expr source, Expr dest)
    {
        if (SuppressedPatterns.TryGetValue(source, out var srcPatterns))
        {
            if (!SuppressedPatterns.TryGetValue(dest, out var destPatterns))
            {
                destPatterns = new HashSet<IPattern>(srcPatterns);
                SuppressedPatterns.Add(dest, destPatterns);
            }
            else
            {
                foreach (var pattern in srcPatterns)
                {
                    destPatterns.Add(pattern);
                }
            }
        }
    }

    /// <summary>
    /// Memo rewrite.
    /// </summary>
    /// <param name="from">Source expression.</param>
    /// <param name="to">Dest expression.</param>
    public void MemoRewrite(Expr from, Expr to)
    {
        RewriteMemo[from] = to;
    }

    /// <summary>
    /// Try update expression with rewrite memo.
    /// </summary>
    /// <param name="expr">Expr ref.</param>
    /// <returns>Has rewrited.</returns>
    public bool TryUpdateWithRewrite(ref Expr expr)
    {
        if (RewriteMemo.TryGetValue(expr, out var newExpr))
        {
            expr = newExpr;
            return true;
        }

        return false;
    }
}
