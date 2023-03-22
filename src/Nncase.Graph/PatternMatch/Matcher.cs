// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Reactive;
using System.Reactive.Joins;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.Utilities;
using Tensorflow.Contexts;

namespace Nncase.PatternMatch;

internal sealed partial class Matcher : ExprFunctor<bool, Unit, IPattern>
{
    private readonly MatchOptions _options;
    private MatchScope _currentScope = new MatchScope();

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
        matcher.Visit(expr, pattern);
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

    protected override bool VisitConst(Const expr, IPattern pattern)
    {
        if (pattern is ConstPattern constPattern)
        {
            return constPattern.MatchLeaf(expr);
        }

        return DefaultVisit(expr, pattern);
    }

    protected override bool VisitOp(Op expr, IPattern pattern)
    {
        if (pattern is IOpPattern opPattern)
        {
            return opPattern.MatchLeaf(expr);
        }

        return DefaultVisit(expr, pattern);
    }

    protected override bool DispatchVisit(Expr expr, IPattern pattern)
    {
        bool isMatch = _currentScope.IsMatch;
        if (isMatch)
        {
            if (_options.IsSuppressedPattern(expr, pattern))
            {
                isMatch = false;
            }
            else if (_currentScope.TryGetMemo(pattern, out var oldExpr))
            {
                if (!ReferenceEquals(oldExpr, expr))
                {
                    isMatch = false;
                }
            }
            else
            {
                if (expr.Accept(this, pattern))
                {
                    _currentScope.AddMatch(pattern, expr);
                }
                else
                {
                    isMatch = false;
                }
            }

            if (!isMatch)
            {
                _currentScope.IsMatch = false;
            }
        }

        return isMatch;
    }

    protected override bool DefaultVisit(Expr expr, IPattern pattern)
    {
        return pattern switch
        {
            OrPattern orPattern => VisitOrPattern(expr, orPattern),
            ExprPattern exprPattern => VisitExprPattern(expr, exprPattern),
            _ => false,
        };
    }

    private bool VisitOrPattern(Expr expr, OrPattern orPattern)
    {
        // Preserve context
        var oldScope = _currentScope;
        _currentScope = new MatchScope(oldScope);

        if (!Visit(expr, orPattern.ConditionA))
        {
            // Try plan B
            _currentScope = new MatchScope(oldScope);
            return Visit(expr, orPattern.ConditionB);
        }

        return true;
    }

    private bool VisitExprPattern(Expr expr, ExprPattern exprPattern)
    {
        return exprPattern.MatchLeaf(expr);
    }

    private bool VisitVArgsPattern<T>(ReadOnlySpan<T> exprs, VArgsPattern vArgsPattern)
        where T : Expr
    {
        bool isMatch = vArgsPattern.MatchLeaf(SpanUtility.UnsafeCast<T, Expr>(exprs));
        if (isMatch)
        {
            for (int i = 0; i < exprs.Length; i++)
            {
                isMatch = Visit(exprs[i], vArgsPattern.Fields[i]);
                if (!isMatch)
                {
                    break;
                }
            }
        }

        if (isMatch)
        {
            _currentScope.AddMatch(vArgsPattern, exprs.ToArray());
            return true;
        }
        else
        {
            _currentScope.IsMatch = false;
            return false;
        }
    }

    private sealed class MatchVisitor : ExprWalker
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

        protected override Unit DefaultVisitLeaf(Expr expr)
        {
            if (!_options.IsSuppressedPattern(expr, _rootPattern) && _rootPattern.MatchLeaf(expr))
            {
                _candidates.Add(expr);
            }

            return default;
        }
    }
}
