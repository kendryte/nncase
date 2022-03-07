// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using LanguageExt;
using Nncase.IR;
using Nncase.Transform;

namespace Nncase.PatternMatch;

internal sealed class EGraphMatcher
{
    private readonly IReadOnlyDictionary<EClass, List<ENode>> _eclasses;

    public EGraphMatcher(IReadOnlyDictionary<EClass, List<ENode>> eclasses)
    {
        _eclasses = eclasses;
    }

    /// <summary>
    /// Match expression as root.
    /// </summary>
    /// <param name="eclasses">EClasses.</param>
    /// <param name="pattern">Pattern.</param>
    /// <param name="enode">ENode.</param>
    /// <param name="results">Match results.</param>
    /// <returns>Match success.</returns>
    public static bool TryMatchRoot(IReadOnlyDictionary<EClass, List<ENode>> eclasses, IPattern pattern, ENode enode, [MaybeNullWhen(false)] out IReadOnlyList<IMatchResult> results)
    {
        if (!pattern.MatchLeaf(enode.Expr))
        {
            results = null;
            return false;
        }

        var rootScopes = new[] { new MatchScope() };
        var matcher = new EGraphMatcher(eclasses);
        var matchScopes = matcher.Visit(rootScopes, pattern, enode);
        if (matchScopes.Count == 0)
        {
            results = null;
            return false;
        }
        else
        {
            results = matchScopes.Select(x =>
              {
                  x.TryGetMatchResult(out var result);
                  return result!;
              }).ToList();
            return results.Count > 0;
        }
    }

    /// <summary>
    /// Match expression.
    /// </summary>
    /// <param name="eclasses">EClasses.</param>
    /// <param name="pattern">Pattern.</param>
    /// <param name="enodes">ENodes.</param>
    /// <param name="results">Match results.</param>
    /// <returns>Match success.</returns>
    public static bool TryMatch(IReadOnlyDictionary<EClass, List<ENode>> eclasses, IPattern pattern, IReadOnlyList<ENode> enodes, [MaybeNullWhen(false)] out IReadOnlyList<IMatchResult> results)
    {
        var rootScopes = new[] { new MatchScope() };
        var matcher = new EGraphMatcher(eclasses);
        var matchScopes = new List<MatchScope>();

        foreach (var enode in enodes)
        {
            var scopes = matcher.Visit(rootScopes, pattern, enode);
            matchScopes.AddRange(scopes);
        }

        if (matchScopes.Count == 0)
        {
            results = null;
            return false;
        }
        else
        {
            results = matchScopes.Select(x =>
            {
                x.TryGetMatchResult(out var result);
                return result!;
            }).ToList();
            return results.Count > 0;
        }
    }

    private IReadOnlyList<MatchScope> Visit(IReadOnlyList<MatchScope> matchScopes, IPattern pattern, ENode enode)
    {
        return (pattern, enode.Expr) switch
        {
            (VarPattern varPat, Var var) => VisitLeaf(matchScopes, varPat, enode, var),
            (TensorConstPattern constPat, TensorConst con) => VisitLeaf(matchScopes, constPat, enode, con),
            (TupleConstPattern constPat, TupleConst con) => VisitLeaf(matchScopes, constPat, enode, con),
            (ConstPattern constPat, Const con) => VisitLeaf(matchScopes, constPat, enode, con),
            (FunctionPattern functionPat, Function func) => Visit(matchScopes, functionPat, enode, func),
            (CallPattern callPat, Call call) => Visit(matchScopes, callPat, enode, call),
            (TuplePattern tuplePat, IR.Tuple tuple) => Visit(matchScopes, tuplePat, enode, tuple),
            (IOpPattern opPat, Op op) => VisitLeaf(matchScopes, opPat, enode, op),
            (OrPattern orPat, _) => Visit(matchScopes, orPat, enode, enode.Expr),
            (ExprPattern exprPattern, Expr expr) => VisitLeaf(matchScopes, exprPattern, enode, expr),
            _ => Array.Empty<MatchScope>(),
        };
    }

    private IReadOnlyList<MatchScope> Visit(IReadOnlyList<MatchScope> matchScopes, IPattern pattern, EClass eClass)
    {
        var newScopes = new List<MatchScope>();

        foreach (var node in _eclasses[eClass])
        {
            var scopes = Visit(matchScopes, pattern, node);
            if (scopes.Count > 0)
            {
                newScopes.AddRange(newScopes);
            }
        }

        return newScopes;
    }

    private IReadOnlyList<MatchScope> VisitLeaf(IReadOnlyList<MatchScope> matchScopes, IPattern pattern, ENode enode, Expr expr)
    {
        var context = new MatchContext(matchScopes, pattern, expr);

        if (context.HasCandidates
            && pattern.MatchLeaf(expr))
        {
            context.NewScopes.AddRange(context.Candidates);
            context.MatchCandidates(pattern, expr);
        }

        return context.NewScopes;
    }

    private IReadOnlyList<MatchScope> Visit(IReadOnlyList<MatchScope> matchScopes, FunctionPattern pattern, ENode enode, Function expr)
    {
        var context = new MatchContext(matchScopes, pattern, expr);

        if (context.HasCandidates
            && pattern.MatchLeaf(expr))
        {
            var newScopes = Visit(context.Candidates, pattern.Body, enode.Children[0]);
            newScopes = Visit(newScopes, pattern.Parameters, enode.Children.Skip(1));

            if (newScopes.Count > 0)
            {
                context.NewScopes.AddRange(newScopes);
                context.MatchCandidates(pattern, expr);
            }
        }

        return context.NewScopes;
    }

    private IReadOnlyList<MatchScope> Visit(IReadOnlyList<MatchScope> matchScopes, CallPattern pattern, ENode enode, Call expr)
    {
        var context = new MatchContext(matchScopes, pattern, expr);

        if (context.HasCandidates
            && pattern.MatchLeaf(expr))
        {
            var newScopes = Visit(context.Candidates, pattern.Target, enode.Children[0]);
            newScopes = Visit(newScopes, pattern.Parameters, enode.Children.Skip(1));

            if (newScopes.Count > 0)
            {
                context.NewScopes.AddRange(newScopes);
                context.MatchCandidates(pattern, expr);
            }
        }

        return context.NewScopes;
    }

    private IReadOnlyList<MatchScope> Visit(IReadOnlyList<MatchScope> matchScopes, TuplePattern pattern, ENode enode, IR.Tuple expr)
    {
        var context = new MatchContext(matchScopes, pattern, expr);

        if (context.HasCandidates
            && pattern.MatchLeaf(expr))
        {
            var newScopes = Visit(context.Candidates, pattern.Fields, enode.Children);

            if (newScopes.Count > 0)
            {
                context.NewScopes.AddRange(newScopes);
                context.MatchCandidates(pattern, expr);
            }
        }

        return context.NewScopes;
    }

    private IReadOnlyList<MatchScope> Visit(IReadOnlyList<MatchScope> matchScopes, OrPattern pattern, ENode enode, Expr expr)
    {
        var context = new MatchContext(matchScopes, pattern, expr);

        if (context.HasCandidates
            && pattern.MatchLeaf(expr))
        {
            var scopesA = Visit(context.Candidates, pattern.ConditionA, enode);
            var scopesB = Visit(context.Candidates, pattern.ConditionB, enode);

            if (scopesA.Count > 0)
            {
                context.NewScopes.AddRange(scopesA);
            }

            if (scopesB.Count > 0)
            {
                context.NewScopes.AddRange(scopesB);
            }

            if (scopesA.Count > 0 || scopesB.Count > 0)
            {
                context.MatchCandidates(pattern, expr);
            }
        }

        return context.NewScopes;
    }

    private IReadOnlyList<MatchScope> Visit(IReadOnlyList<MatchScope> matchScopes, VArgsPattern pattern, IReadOnlyList<ENode> enodes)
    {
        var exprs = enodes.Select(x => x.Expr).ToList();
        var context = new MatchContext(matchScopes, pattern, exprs);

        if (context.HasCandidates
            && pattern.MatchLeaf(exprs))
        {
            IReadOnlyList<MatchScope> scopes = context.Candidates;
            for (int i = 0; i < pattern.Count; i++)
            {
                scopes = Visit(scopes, pattern[i], enodes[i]);
                if (scopes.Count == 0)
                {
                    break;
                }
            }

            if (scopes.Count > 0)
            {
                context.NewScopes.AddRange(scopes);
                context.MatchCandidates(pattern, exprs);
            }
        }

        return context.NewScopes;
    }

    private IReadOnlyList<MatchScope> Visit(IReadOnlyList<MatchScope> matchScopes, VArgsPattern pattern, IEnumerable<EClass> eClasses)
    {
        if (eClasses.Count() != pattern.Count)
        {
            return Array.Empty<MatchScope>();
        }
        else
        {
            var newScopes = new List<MatchScope>();

            foreach (var enodes in (from ec in eClasses
                                    select from en in _eclasses[ec]
                                           select en).CartesianProduct())
            {
                var scopes = Visit(matchScopes, pattern, enodes.ToList());
                if (scopes.Count() > 0)
                {
                    newScopes.AddRange(scopes);
                }
            }

            return newScopes;
        }
    }
}
