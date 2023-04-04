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
    }

    /// <summary>
    /// Gets suppressed patterns.
    /// </summary>
    public Dictionary<Expr, HashSet<IPattern>> SuppressedPatterns { get; }

    /// <summary>
    /// check the expr and pattern in the suppressed pattern dict.
    /// </summary>
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
    public void SuppressPattern(Expr expr, IPattern pattern)
    {
        if (!SuppressedPatterns.TryGetValue(expr, out var patterns))
        {
            patterns = new HashSet<IPattern>(ReferenceEqualityComparer.Instance);
            SuppressedPatterns.Add(expr, patterns);
        }

        patterns.Add(pattern);
    }

    /// <summary>
    /// when the soure expr has been changed, need inherit is suppress attribute into new expr.
    /// </summary>
    public void InheritSuppressPatterns(Expr source, Expr dest)
    {
        if (ReferenceEquals(source, dest))
        {
            return;
        }

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
}
