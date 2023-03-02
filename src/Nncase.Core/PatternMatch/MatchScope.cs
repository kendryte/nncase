// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.Passes;

namespace Nncase.PatternMatch;

/// <summary>
/// Match scope.
/// </summary>
public sealed class MatchScope
{
    private readonly object? _root;
    private readonly MatchScope? _parent;
    private readonly Dictionary<IPattern, Expr> _patMemo = new(ReferenceEqualityComparer.Instance);
    private readonly Dictionary<VArgsPattern, IReadOnlyList<Expr>> _vargspatMemo = new(ReferenceEqualityComparer.Instance);
    private readonly List<(IPattern Pattern, object Match)> _matches = new();

    /// <summary>
    /// Initializes a new instance of the <see cref="MatchScope"/> class.
    /// </summary>
    /// <param name="root">Match root.</param>
    public MatchScope(Expr root)
    {
        _root = root;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="MatchScope"/> class.
    /// </summary>
    /// <param name="root">Match root.</param>
    public MatchScope(ENode root)
    {
        _root = root;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="MatchScope"/> class.
    /// </summary>
    /// <param name="parent">Parent scope.</param>
    public MatchScope(MatchScope? parent = null)
    {
        _parent = parent;
    }

    /// <summary>
    /// Gets or sets a value indicating whether is match.
    /// </summary>
    public bool IsMatch { get; set; } = true;

    /// <summary>
    /// Create a new scope inherting this scope.
    /// </summary>
    /// <returns>New scope.</returns>
    public MatchScope BeginScope()
    {
        var scope = new MatchScope(this);
        return scope;
    }

    /// <summary>
    /// Try get memo.
    /// </summary>
    /// <param name="pattern">Pattern.</param>
    /// <param name="expr">Expression.</param>
    /// <returns>Operation succeeded.</returns>
    public bool TryGetMemo(IPattern pattern, [MaybeNullWhen(false)] out Expr expr)
    {
        if (_patMemo.TryGetValue(pattern, out expr))
        {
            return true;
        }
        else if (_parent != null)
        {
            return _parent.TryGetMemo(pattern, out expr);
        }

        expr = null;
        return false;
    }

    /// <summary>
    /// Try get memo.
    /// </summary>
    /// <param name="pattern">Pattern.</param>
    /// <param name="exprs">Expressions.</param>
    /// <returns>Operation succeeded.</returns>
    public bool TryGetMemo(VArgsPattern pattern, [MaybeNullWhen(false)] out IReadOnlyList<Expr> exprs)
    {
        if (_vargspatMemo.TryGetValue(pattern, out exprs))
        {
            return true;
        }
        else if (_parent != null)
        {
            return _parent.TryGetMemo(pattern, out exprs);
        }

        exprs = null;
        return false;
    }

    /// <summary>
    /// Add match.
    /// </summary>
    /// <param name="pattern">Pattern.</param>
    /// <param name="match">Match expression.</param>
    public void AddMatch(IPattern pattern, Expr match)
    {
        _patMemo.Add(pattern, match);
        _matches.Add((pattern, match));
    }

    /// <summary>
    /// Add match.
    /// </summary>
    /// <param name="pattern">Pattern.</param>
    /// <param name="match">Match expressions.</param>
    public void AddMatch(VArgsPattern pattern, IReadOnlyList<Expr> match)
    {
        _vargspatMemo.Add(pattern, match);
        _matches.Add((pattern, match));
    }

    /// <summary>
    /// Get flatten match result.
    /// </summary>
    /// <param name="result">Match result.</param>
    /// <returns>Match success.</returns>
    public bool TryGetMatchResult([MaybeNullWhen(false)] out IMatchResult result)
    {
        if (IsMatch)
        {
            var matches = new Dictionary<IPattern, object>(ReferenceEqualityComparer.Instance);
            var root = FlattenMatches(matches);
            result = new MatchResult(root, matches);
            return true;
        }
        else
        {
            result = null;
            return false;
        }
    }

    private object FlattenMatches(Dictionary<IPattern, object> matches)
    {
        object root;
        if (_parent != null)
        {
            root = _parent.FlattenMatches(matches);
        }
        else
        {
            root = _root!;
        }

        foreach (var match in _matches)
        {
            matches.Add(match.Pattern, match.Match);
        }

        return root;
    }
}
