// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.Passes;

namespace Nncase.PatternMatch;

/// <summary>
/// egraph matcher.
/// </summary>
public sealed class EGraphMatcher
{
    /// <summary>
    /// Match enodes as root.
    /// </summary>
    /// <param name="enodes">ENodes.</param>
    /// <param name="pattern">Pattern.</param>
    /// <param name="results">Match results.</param>
    /// <returns>Match success.</returns>
    public static bool TryMatchRoot(IEnumerable<ENode> enodes, IPattern pattern, [MaybeNullWhen(false)] out IReadOnlyList<IMatchResult> results)
    {
        var matcher = new EGraphMatcher();
        var matchScopes = new List<MatchScope>();

        foreach (var enode in enodes)
        {
            if (pattern.MatchLeaf(enode.Expr))
            {
                var scopes = matcher.Visit(new[] { new MatchScope(enode) }, pattern, enode);
                matchScopes.AddRange(scopes);
            }
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
            (FusionPattern fusionPattern, Fusion fusion) => VisitLeaf(matchScopes, fusionPattern, enode, fusion),
            (FunctionPattern functionPat, Function func) => Visit(matchScopes, functionPat, enode, func),
            (CallPattern callPat, Call call) => Visit(matchScopes, callPat, enode, call),
            (MarkerPattern mkPat, Marker mk) => Visit(matchScopes, mkPat, enode, mk),
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

        foreach (var node in eClass.Nodes)
        {
            var scopes = Visit(matchScopes, pattern, node);
            if (scopes.Count > 0)
            {
                newScopes.AddRange(scopes);
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
            if (newScopes.Count > 0)
            {
                newScopes = Visit(newScopes, pattern.Parameters, enode.Children.Skip(1));
                if (newScopes.Count > 0)
                {
                    context.NewScopes.AddRange(newScopes);
                    context.MatchCandidates(pattern, expr);
                }
            }
        }

        return context.NewScopes;
    }

    private IReadOnlyList<MatchScope> VisitLeaf(IReadOnlyList<MatchScope> matchScopes, FusionPattern pattern, ENode enode, Fusion expr)
    {
        var context = new MatchContext(matchScopes, pattern, expr);

        if (context.HasCandidates
            && CompilerServices.TryMatchRoot(expr, pattern, out var result))
        {
            context.NewScopes.AddRange(context.Candidates);
            context.MatchCandidates(pattern, (Expr)result[pattern]);
        }

        return context.NewScopes;
    }

    private IReadOnlyList<MatchScope> Visit(IReadOnlyList<MatchScope> matchScopes, CallPattern pattern, ENode enode, Call expr)
    {
        var context = new MatchContext(matchScopes, pattern, expr);

        if (context.HasCandidates
            && pattern.MatchLeaf(expr)
            && pattern.Target.MatchLeaf(expr.Target)
            && pattern.Arguments.MatchLeaf(expr.Arguments))
        {
            var newScopes = Visit(context.Candidates, pattern.Target, enode.Children[0]);
            if (newScopes.Count > 0)
            {
                newScopes = Visit(newScopes, pattern.Arguments, enode.Children.Skip(1));
                if (newScopes.Count > 0)
                {
                    context.NewScopes.AddRange(newScopes);
                    context.MatchCandidates(pattern, expr);
                }
            }
        }

        return context.NewScopes;
    }

    private IReadOnlyList<MatchScope> Visit(IReadOnlyList<MatchScope> matchScopes, MarkerPattern pattern, ENode enode, Marker expr)
    {
        var context = new MatchContext(matchScopes, pattern, expr);

        if (context.HasCandidates
            && pattern.MatchLeaf(expr))
        {
            var newScopes = Visit(context.Candidates, pattern.Target, enode.Children[0]);
            if (newScopes.Count > 0)
            {
                newScopes = Visit(newScopes, pattern.Attribute, enode.Children[1]);
                if (newScopes.Count > 0)
                {
                    context.NewScopes.AddRange(newScopes);
                    context.MatchCandidates(pattern, expr);
                }
            }
        }

        return context.NewScopes;
    }

    private IReadOnlyList<MatchScope> Visit(IReadOnlyList<MatchScope> matchScopes, TuplePattern pattern, ENode enode, IR.Tuple expr)
    {
        var context = new MatchContext(matchScopes, pattern, expr);

        if (context.HasCandidates
            && pattern.MatchLeaf(expr)
            && pattern.Fields.MatchLeaf(expr.Fields))
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
        var exprs = enodes.Select(x => x.Expr).ToArray();
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
        if (pattern.Count == 0 || eClasses.Count() != pattern.Count)
        {
            return Array.Empty<MatchScope>();
        }
        else
        {
            var newScopes = new List<MatchScope>();

            foreach (var enodes in (from ec in eClasses
                                    select from en in ec.Nodes
                                           select en).CartesianProduct())
            {
                var scopes = Visit(matchScopes, pattern, enodes.ToList());
                if (scopes.Count > 0)
                {
                    newScopes.AddRange(scopes);
                }
            }

            return newScopes;
        }
    }
}
