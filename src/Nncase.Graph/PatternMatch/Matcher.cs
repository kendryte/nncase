// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.PatternMatch;

internal sealed class Matcher
{
    private readonly Dictionary<IPattern, Expr> _patMemo = new(ReferenceEqualityComparer.Instance);
    private readonly Dictionary<VArgsPattern, Expr[]> _vargspatMemo = new(ReferenceEqualityComparer.Instance);

    private readonly List<(IPattern, object)> _matches = new();

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

        var matcher = new Matcher();
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

    private bool Visit(IPattern pattern, Expr expr)
    {
        return (pattern, expr) switch
        {
            (VarPattern varPat, Var var) => Visit(varPat, var),
            (TensorConstPattern constPat, TensorConst con) => Visit(constPat, con),
            (TupleConstPattern constPat, TupleConst con) => Visit(constPat, con),
            (ConstPattern constPat, Const con) => Visit(constPat, con),
            (FunctionPattern functionPat, Function func) => Visit(functionPat, func),
            (CallPattern callPat, Call call) => Visit(callPat, call),
            (TuplePattern tuplePat, IR.Tuple tuple) => Visit(tuplePat, tuple),
            (IOpPattern opPat, Op op) => Visit(opPat, op),
            (IOrPattern orPat, _) => Visit(orPat, expr),
            (ExprPattern exprPattern, _) => Visit(exprPattern, expr),
            _ => false,
        };
    }

    private bool Visit(VarPattern pattern, Var expr)
    {
        if (_patMemo.TryGetValue(pattern, out var oldExpr))
        {
            return ReferenceEquals(oldExpr, expr);
        }
        else
        {
            if (pattern.MatchLeaf(expr))
            {
                _matches.Add((pattern, expr));
                return true;
            }

            return false;
        }
    }

    private bool Visit(TensorConstPattern pattern, TensorConst expr)
    {
        if (_patMemo.TryGetValue(pattern, out var oldExpr))
        {
            return ReferenceEquals(oldExpr, expr);
        }
        else
        {
            if (pattern.MatchLeaf(expr))
            {
                _matches.Add((pattern, expr));
                return true;
            }

            return false;
        }
    }

    private bool Visit(TupleConstPattern pattern, TupleConst expr)
    {
        if (_patMemo.TryGetValue(pattern, out var oldExpr))
        {
            return ReferenceEquals(oldExpr, expr);
        }
        else
        {
            if (pattern.MatchLeaf(expr))
            {
                _matches.Add((pattern, expr));
                return true;
            }

            return false;
        }
    }

    private bool Visit(ConstPattern pattern, Const expr)
    {
        if (_patMemo.TryGetValue(pattern, out var oldExpr))
        {
            return ReferenceEquals(oldExpr, expr);
        }
        else
        {
            if (pattern.MatchLeaf(expr))
            {
                _matches.Add((pattern, expr));
                return true;
            }

            return false;
        }
    }

    private bool Visit(FunctionPattern pattern, Function expr)
    {
        if (_patMemo.TryGetValue(pattern, out var oldExpr))
        {
            return ReferenceEquals(oldExpr, expr);
        }
        else
        {
            if (pattern.MatchLeaf(expr)
                && Visit(pattern.Body, expr.Body)
                && Visit(pattern.Parameters, expr.Parameters))
            {
                _matches.Add((pattern, expr));
                return true;
            }

            return false;
        }
    }

    private bool Visit(CallPattern pattern, Call expr)
    {
        if (_patMemo.TryGetValue(pattern, out var oldExpr))
        {
            return ReferenceEquals(oldExpr, expr);
        }
        else
        {
            if (pattern.MatchLeaf(expr)
                && Visit(pattern.Target, expr.Target)
                && Visit(pattern.Parameters, expr.Parameters))
            {
                _matches.Add((pattern, expr));
                return true;
            }

            return false;
        }
    }

    private bool Visit(TuplePattern pattern, IR.Tuple expr)
    {
        if (_patMemo.TryGetValue(pattern, out var oldExpr))
        {
            return ReferenceEquals(oldExpr, expr);
        }
        else
        {
            if (pattern.MatchLeaf(expr)
                && Visit(pattern.Fields, expr.Fields))
            {
                _matches.Add((pattern, expr));
                return true;
            }

            return false;
        }
    }

    private bool Visit(IOpPattern pattern, Op expr)
    {
        if (_patMemo.TryGetValue(pattern, out var oldExpr))
        {
            return ReferenceEquals(oldExpr, expr);
        }
        else
        {
            if (pattern.MatchLeaf(expr))
            {
                _matches.Add((pattern, expr));
                return true;
            }

            return false;
        }
    }

    private bool Visit(IOrPattern pattern, Expr expr)
    {
        if (_patMemo.TryGetValue(pattern, out var oldExpr))
        {
            return ReferenceEquals(oldExpr, expr);
        }
        else
        {
            if (pattern.MatchLeaf(expr))
            {
                var oldMatches = _matches.Count;
                if (!Visit(pattern.ConditionA, expr))
                {
                    // cleanup old matches
                    if (_matches.Count > oldMatches)
                    {
                        _matches.RemoveRange(oldMatches, _matches.Count - oldMatches);
                    }

                    if (!Visit(pattern.ConditionB, expr))
                    {
                        return false;
                    }
                }

                _matches.Add((pattern, expr));
                return true;
            }

            return false;
        }
    }

    private bool Visit(ExprPattern pattern, Expr expr)
    {
        if (_patMemo.TryGetValue(pattern, out var oldExpr))
        {
            return ReferenceEquals(oldExpr, expr);
        }
        else
        {
            if (pattern.MatchLeaf(expr))
            {
                _matches.Add((pattern, expr));
                return true;
            }

            return false;
        }
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

    private sealed class MatchScope
    {
        private readonly MatchScope? _parent;
        private readonly Dictionary<IPattern, Expr> _patMemo = new(ReferenceEqualityComparer.Instance);
        private readonly Dictionary<VArgsPattern, Expr[]> _vargspatMemo = new(ReferenceEqualityComparer.Instance);
        private readonly List<(IPattern Pattern, object Match)> _matches = new();

        public MatchScope(MatchScope? parent)
        {
            _parent = parent;
        }

        public bool IsMatch { get; set; } = true;

        public MatchScope BeginScope()
        {
            return new MatchScope(this);
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
