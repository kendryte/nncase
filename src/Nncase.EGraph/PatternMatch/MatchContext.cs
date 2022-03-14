// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.PatternMatch;

internal enum MemoFilterStatus
{
    /// <summary>
    /// Expr(s) not found.
    /// </summary>
    NotFound,

    /// <summary>
    /// Expr(s) equals to found.
    /// </summary>
    Equals,

    /// <summary>
    /// Expr(s) not equals to found.
    /// </summary>
    NotEquals,
}

internal struct MatchContext
{
    public MatchContext(IReadOnlyList<MatchScope> matchScopes, Func<MatchScope, MemoFilterStatus> filter)
    {
        foreach (var scope in matchScopes)
        {
            var status = filter(scope);
            switch (status)
            {
                case MemoFilterStatus.NotFound:
                    Candidates.Add(scope.BeginScope());
                    break;
                case MemoFilterStatus.Equals:
                    NewScopes.Add(scope);
                    break;
                case MemoFilterStatus.NotEquals:
                    break;
                default:
                    break;
            }
        }
    }

    public MatchContext(IReadOnlyList<MatchScope> matchScopes, IPattern pattern, Expr expr)
        : this(matchScopes, scope =>
        {
            if (scope.TryGetMemo(pattern, out var oldExpr))
            {
                return object.ReferenceEquals(oldExpr, expr) ? MemoFilterStatus.Equals : MemoFilterStatus.NotEquals;
            }
            else
            {
                return MemoFilterStatus.NotFound;
            }
        })
    {
    }

    public MatchContext(IReadOnlyList<MatchScope> matchScopes, VArgsPattern pattern, IReadOnlyList<Expr> exprs)
        : this(matchScopes, scope =>
        {
            if (scope.TryGetMemo(pattern, out var oldExprs))
            {
                return oldExprs.SequenceEqual(exprs, ReferenceEqualityComparer.Instance)
                    ? MemoFilterStatus.Equals : MemoFilterStatus.NotEquals;
            }
            else
            {
                return MemoFilterStatus.NotFound;
            }
        })
    {
    }

    public List<MatchScope> NewScopes { get; } = new();

    public List<MatchScope> Candidates { get; set; } = new();

    public bool HasCandidates => Candidates.Count > 0;

    public void MatchCandidates(IPattern pattern, Expr expr)
    {
        if (HasCandidates)
        {
            Candidates.ForEach(x => x.AddMatch(pattern, expr));
        }
    }

    public void MatchCandidates(VArgsPattern pattern, IReadOnlyList<Expr> exprs)
    {
        if (HasCandidates)
        {
            Candidates.ForEach(x => x.AddMatch(pattern, exprs));
        }
    }
}
