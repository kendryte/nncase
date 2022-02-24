// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
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
    private readonly List<MatchContext> _contexts = new();

    public EGraphMatcher(IReadOnlyDictionary<EClass, List<ENode>> eclasses)
    {
        _eclasses = eclasses;
    }

    /// <summary>
    /// Match expression as root.
    /// </summary>
    /// <param name="pattern">Pattern.</param>
    /// <param name="expr">Expression.</param>
    /// <returns>Match result.</returns>
    public static IMatchResult? MatchRoot(IPattern pattern, Expr expr)
    {
        if (!pattern.MatchLeaf(expr))
        {
            return null;
        }

        var matcher = new EGraphMatcher();
        return matcher.Visit(pattern, expr)
            ? new MatchResult(matcher._matches.ToDictionary(x => x.Item1, x => x.Item2, (IEqualityComparer<IPattern?>)ReferenceEqualityComparer.Instance))
            : null;
    }

    /// <summary>
    /// Match expression.
    /// </summary>
    /// <param name="pattern">Pattern.</param>
    /// <param name="expr">Expression.</param>
    /// <returns>Match result.</returns>
    public static IMatchResult? Match(IPattern pattern, Expr expr)
    {
        var candidates = new List<Expr>();
        new MatchVisitor(candidates, pattern).Visit(expr);

        foreach (var candidate in candidates)
        {
            var result = MatchRoot(pattern, candidate);
            if (result != null)
            {
                return result;
            }
        }

        return null;
    }

    private List<MatchContext> TidyContexts()
    {
        _contexts.RemoveAll(x => !x.IsMatch);
        return _contexts;
    }

    private bool Visit(IPattern pattern, ENode enode)
    {
        return (pattern, enode.Expr) switch
        {
            (VarPattern varPat, Var var) => VisitLeaf(varPat, enode, var),
            (TensorConstPattern constPat, TensorConst con) => VisitLeaf(constPat, enode, con),
            (TupleConstPattern constPat, TupleConst con) => VisitLeaf(constPat, enode, con),
            (ConstPattern constPat, Const con) => VisitLeaf(constPat, enode, con),
            (FunctionPattern functionPat, Function func) => Visit(functionPat, enode, func),
            (CallPattern callPat, Call call) => Visit(callPat, enode, call),
            (TuplePattern tuplePat, IR.Tuple tuple) => Visit(tuplePat, enode, tuple),
            (IOpPattern opPat, Op op) => VisitLeaf(opPat, enode, op),
            (IOrPattern orPat, _) => Visit(orPat, enode, expr),
            (ExprPattern exprPattern, Expr expr) => VisitLeaf(exprPattern, enode, expr),
            _ => false,
        };
    }

    private bool Visit(IPattern pattern, EClass eClass)
    {
        bool match = false;

        foreach (var node in _eclasses[eClass])
        {
            if (Visit(pattern, node))
            {
                match = true;
            }
        }

        return match;
    }

    private bool Visit(VArgsPattern pattern, IEnumerable<EClass> eClasses)
    {
        bool match = false;

        foreach (var node in _eclasses[eClass])
        {
            if (Visit(pattern, node))
            {
                match = true;
            }
        }

        return match;
    }

    private bool VisitLeaf(IPattern pattern, ENode enode, Expr expr)
    {
        bool match = false;

        foreach (var context in TidyContexts())
        {
            if (context.PatMemo.TryGetValue(pattern, out var oldExpr))
            {
                if (ReferenceEquals(oldExpr, expr))
                {
                    match = true;
                }
                else
                {
                    context.IsMatch = false;
                }
            }
            else
            {
                if (pattern.MatchLeaf(expr))
                {
                    context.Matches.Add((pattern, expr));
                    match = true;
                }
                else
                {
                    context.IsMatch = false;
                }
            }
        }

        return match;
    }

    private bool Visit(FunctionPattern pattern, ENode enode, Function expr)
    {
        bool match = false;

        foreach (var context in TidyContexts().ToArray())
        {
            if (context.PatMemo.TryGetValue(pattern, out var oldExpr))
            {
                if (ReferenceEquals(oldExpr, expr))
                {
                    match = true;
                }
                else
                {
                    context.IsMatch = false;
                }
            }
            else
            {
                if (pattern.MatchLeaf(expr)
                    && Visit(pattern.Body, enode.Children[0])
                    && Visit(pattern.Parameters, enode.Children.Skip(1)))
                {
                    context.Matches.Add((pattern, expr));
                    match = true;
                }
                else
                {
                    context.IsMatch = false;
                }
            }
        }

        return match;
    }

    private bool Visit(CallPattern pattern, ENode enode, Call expr)
    {
        bool match = false;

        foreach (var context in TidyContexts().ToArray())
        {
            if (context.PatMemo.TryGetValue(pattern, out var oldExpr))
            {
                if (ReferenceEquals(oldExpr, expr))
                {
                    match = true;
                }
                else
                {
                    context.IsMatch = false;
                }
            }
            else
            {
                if (pattern.MatchLeaf(expr)
                    && Visit(pattern.Target, enode.Children[0])
                    && Visit(pattern.Parameters, enode.Children.Skip(1)))
                {
                    context.Matches.Add((pattern, expr));
                    match = true;
                }
                else
                {
                    context.IsMatch = false;
                }
            }
        }

        return match;
    }

    private bool Visit(TuplePattern pattern, ENode enode, IR.Tuple expr)
    {
        bool match = false;

        foreach (var context in TidyContexts().ToArray())
        {
            if (context.PatMemo.TryGetValue(pattern, out var oldExpr))
            {
                if (ReferenceEquals(oldExpr, expr))
                {
                    match = true;
                }
                else
                {
                    context.IsMatch = false;
                }
            }
            else
            {
                if (pattern.MatchLeaf(expr)
                    && Visit(pattern.Fields, enode.Children[0]))
                {
                    context.Matches.Add((pattern, expr));
                    match = true;
                }
                else
                {
                    context.IsMatch = false;
                }
            }
        }

        return match;
    }

    private bool Visit(IOrPattern pattern, ENode enode, Expr expr)
    {
        bool match = false;

        foreach (var context in TidyContexts().ToArray())
        {
            if (context.PatMemo.TryGetValue(pattern, out var oldExpr))
            {
                if (ReferenceEquals(oldExpr, expr))
                {
                    match = true;
                }
                else
                {
                    context.IsMatch = false;
                }
            }
            else
            {
                if (pattern.MatchLeaf(expr))
                {
                    var oldMatches = context.Matches.Count;
                    if (!pattern.ConditionA.MatchLeaf(expr))
                    {
                        // cleanup old matches
                        if (context.Matches.Count > oldMatches)
                        {
                            context.Matches.RemoveRange(oldMatches, context.Matches.Count - oldMatches);
                        }

                        if (!pattern.ConditionB.MatchLeaf(expr))
                        {
                            context.IsMatch = false;
                        }
                    }

                    if (context.IsMatch)
                    {
                        context.Matches.Add((pattern, expr));
                        match = true;
                    }
                }
            }
        }

        return match;
    }

    private bool Visit(VArgsPattern pattern, IReadOnlyList<Expr> exprs)
    {
        if (_vargspatMemo.TryGetValue(pattern, out var oldExprs))
        {
            return oldExprs.SequenceEqual(exprs, ReferenceEqualityComparer.Instance);
        }
        else
        {
            if (pattern.MatchLeaf(exprs))
            {
                for (int i = 0; i < pattern.Count; i++)
                {
                    if (!Visit(pattern[i], exprs[i]))
                    {
                        return false;
                    }
                }

                return true;
            }

            return false;
        }
    }

    private class MatchContext
    {
        public Dictionary<IPattern, Expr> PatMemo { get; init; } = new(ReferenceEqualityComparer.Instance);

        public Dictionary<VArgsPattern, Expr[]> VargspatMemo { get; init; } = new(ReferenceEqualityComparer.Instance);

        public List<(IPattern, object)> Matches { get; init; } = new();

        public bool IsMatch { get; set; } = true;

        public MatchContext Clone()
        {
            return new MatchContext
            {
                PatMemo = new(PatMemo),
                VargspatMemo = new(VargspatMemo),
                Matches = new(Matches),
            };
        }
    }

    private sealed class MatchVisitor : ExprVisitor<Expr, IRType>
    {
        private readonly List<Expr> _candidates;
        private readonly IPattern _rootPattern;

        public MatchVisitor(List<Expr> candidates, IPattern rootPattern)
        {
            _candidates = candidates;
            _rootPattern = rootPattern;
        }

        public override Expr DefaultVisitLeaf(Expr expr)
        {
            if (_rootPattern.MatchLeaf(expr))
            {
                _candidates.Add(expr);
            }

            return expr;
        }
    }
}
