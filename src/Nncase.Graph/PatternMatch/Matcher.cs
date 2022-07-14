// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.PatternMatch;

internal sealed class Matcher
{
    private MatchScope _currentScope = new MatchScope();
    private MatchOptions _options;

    private Matcher(Expr root, MatchOptions options)
    {
        _currentScope = new MatchScope(root);
        _options = options;
    }

    /// <summary>
    /// Match expression as root.
    /// </summary>
    /// <param name="expr">Expression.</param>
    /// <param name="pattern">Pattern.</param>
    /// <param name="options">Match options.</param>
    /// <param name="result">Match result.</param>
    /// <returns>Match success.</returns>
    public static bool TryMatchRoot(Expr expr, IPattern pattern, MatchOptions options, [MaybeNullWhen(false)] out IMatchResult result)
    {
        if (options.IsSuppressedPattern(expr, pattern) || !pattern.MatchLeaf(expr))
        {
            result = null;
            return false;
        }

        var matcher = new Matcher(expr, options);
        matcher.Visit(pattern, expr);
        return matcher._currentScope.TryGetMatchResult(out result);
    }

    /// <summary>
    /// Match expression.
    /// </summary>
    /// <param name="expr">Expression.</param>
    /// <param name="pattern">Pattern.</param>
    /// <param name="options">Match options.</param>
    /// <param name="result">Match result.</param>
    /// <returns>Match success.</returns>
    public static bool TryMatch(Expr expr, IPattern pattern, MatchOptions options, [MaybeNullWhen(false)] out IMatchResult result)
    {
        var candidates = new List<Expr>();
        new MatchVisitor(candidates, pattern, options).Visit(expr);

        foreach (var candidate in candidates)
        {
            if (TryMatchRoot(candidate, pattern, options, out result))
            {
                return true;
            }
        }

        result = null;
        return false;
    }

    private bool Visit(IPattern pattern, Expr expr)
    {
        _currentScope.IsMatch = (pattern, expr) switch
        {
            (VarPattern varPat, Var var) => VisitLeaf(varPat, var),
            (TensorConstPattern constPat, TensorConst con) => VisitLeaf(constPat, con),
            (TupleConstPattern constPat, TupleConst con) => VisitLeaf(constPat, con),
            (ConstPattern constPat, Const con) => VisitLeaf(constPat, con),
            (FunctionPattern functionPat, Function func) => Visit(functionPat, func),
            (CallPattern callPat, Call call) => Visit(callPat, call),
            (TuplePattern tuplePat, IR.Tuple tuple) => Visit(tuplePat, tuple),
            (IOpPattern opPat, Op op) => VisitLeaf(opPat, op),
            (MarkerPattern mkPat, Marker mk) => Visit(mkPat, mk),
            (OrPattern orPat, _) => Visit(orPat, expr),
            (ExprPattern exprPattern, _) => VisitLeaf(exprPattern, expr),
            _ => false,
        };
        return _currentScope.IsMatch;
    }

    private bool VisitLeaf(IPattern pattern, Expr expr)
    {
        if (_currentScope.TryGetMemo(pattern, out var oldExpr))
        {
            if (!ReferenceEquals(oldExpr, expr))
            {
                _currentScope.IsMatch = false;
            }
        }
        else
        {
            if (!_options.IsSuppressedPattern(expr, pattern)
                && pattern.MatchLeaf(expr))
            {
                _currentScope.AddMatch(pattern, expr);
            }
            else
            {
                _currentScope.IsMatch = false;
            }
        }

        return _currentScope.IsMatch;
    }

    private bool Visit(FunctionPattern pattern, Function expr)
    {
        if (_currentScope.TryGetMemo(pattern, out var oldExpr))
        {
            if (!ReferenceEquals(oldExpr, expr))
            {
                _currentScope.IsMatch = false;
            }
        }
        else
        {
            if (!_options.IsSuppressedPattern(expr, pattern)
                && pattern.MatchLeaf(expr)
                && Visit(pattern.Body, expr.Body)
                && Visit(pattern.Parameters, expr.Parameters))
            {
                _currentScope.AddMatch(pattern, expr);
            }
            else
            {
                _currentScope.IsMatch = false;
            }
        }

        return _currentScope.IsMatch;
    }

    private bool Visit(CallPattern pattern, Call expr)
    {
        if (_currentScope.TryGetMemo(pattern, out var oldExpr))
        {
            if (!ReferenceEquals(oldExpr, expr))
            {
                _currentScope.IsMatch = false;
            }
        }
        else
        {
            if (!_options.IsSuppressedPattern(expr, pattern)
                && pattern.MatchLeaf(expr)
                && Visit(pattern.Target, expr.Target)
                && Visit(pattern.Parameters, expr.Parameters))
            {
                _currentScope.AddMatch(pattern, expr);
            }
            else
            {
                _currentScope.IsMatch = false;
            }
        }

        return _currentScope.IsMatch;
    }

    private bool Visit(MarkerPattern pattern, Marker expr)
    {
        if (_currentScope.TryGetMemo(pattern, out var oldExpr))
        {
            if (!ReferenceEquals(oldExpr, expr))
            {
                _currentScope.IsMatch = false;
            }
        }
        else
        {
            if (!_options.IsSuppressedPattern(expr, pattern)
                && pattern.MatchLeaf(expr)
                && Visit(pattern.Target, expr.Target)
                && Visit(pattern.Attribute, expr.Attribute))
            {
                _currentScope.AddMatch(pattern, expr);
            }
            else
            {
                _currentScope.IsMatch = false;
            }
        }

        return _currentScope.IsMatch;
    }

    private bool Visit(TuplePattern pattern, IR.Tuple expr)
    {
        if (_currentScope.TryGetMemo(pattern, out var oldExpr))
        {
            if (!ReferenceEquals(oldExpr, expr))
            {
                _currentScope.IsMatch = false;
            }
        }
        else
        {
            if (!_options.IsSuppressedPattern(expr, pattern)
                && pattern.MatchLeaf(expr)
                && Visit(pattern.Fields, expr.Fields))
            {
                _currentScope.AddMatch(pattern, expr);
            }
            else
            {
                _currentScope.IsMatch = false;
            }
        }

        return _currentScope.IsMatch;
    }

    private bool Visit(OrPattern pattern, Expr expr)
    {
        if (_currentScope.TryGetMemo(pattern, out var oldExpr))
        {
            if (!ReferenceEquals(oldExpr, expr))
            {
                _currentScope.IsMatch = false;
            }
        }
        else
        {
            if (!_options.IsSuppressedPattern(expr, pattern)
                && pattern.MatchLeaf(expr))
            {
                // Preserve context
                var oldScope = _currentScope;
                _currentScope = new MatchScope(oldScope);

                if (!Visit(pattern.ConditionA, expr))
                {
                    // Try plan B
                    _currentScope = new MatchScope(oldScope);
                    if (!Visit(pattern.ConditionB, expr))
                    {
                        _currentScope.IsMatch = false;
                    }
                }

                if (_currentScope.IsMatch)
                {
                    oldScope.AddMatch(pattern, expr);
                }
            }
            else
            {
                _currentScope.IsMatch = false;
            }
        }

        return _currentScope.IsMatch;
    }

    private bool Visit(VArgsPattern pattern, IReadOnlyList<Expr> exprs)
    {
        if (_currentScope.TryGetMemo(pattern, out var oldExprs))
        {
            if (!oldExprs.SequenceEqual(exprs, ReferenceEqualityComparer.Instance))
            {
                _currentScope.IsMatch = false;
            }
        }
        else
        {
            if (pattern.MatchLeaf(exprs))
            {
                for (int i = 0; i < pattern.Count; i++)
                {
                    if (!Visit(pattern[i], exprs[i]))
                    {
                        _currentScope.IsMatch = false;
                        break;
                    }
                }
            }
            else
            {
                _currentScope.IsMatch = false;
            }
        }

        return _currentScope.IsMatch;
    }

    private sealed class MatchVisitor : ExprVisitor<Expr, IRType>
    {
        private readonly List<Expr> _candidates;
        private readonly IPattern _rootPattern;
        private readonly MatchOptions _options;

        public MatchVisitor(List<Expr> candidates, IPattern rootPattern, MatchOptions options)
        {
            _candidates = candidates;
            _rootPattern = rootPattern;
            _options = options;
        }

        public override Expr DefaultVisitLeaf(Expr expr)
        {
            if (!_options.IsSuppressedPattern(expr, _rootPattern) && _rootPattern.MatchLeaf(expr))
            {
                _candidates.Add(expr);
            }

            return expr;
        }
    }
}
