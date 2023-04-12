// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.IR;

namespace Nncase.PatternMatch;

/// <summary>
/// Pattern match result.
/// </summary>
public interface IMatchResult : IEnumerable<KeyValuePair<IPattern, object>>
{
    /// <summary>
    /// Gets root.
    /// </summary>
    object Root { get; }

    /// <summary>
    /// Get match result by name.
    /// </summary>
    /// <param name="name">Pattern name.</param>
    /// <returns>Match result.</returns>
    object this[string name] { get; }

    /// <summary>
    /// Get match result by pattern.
    /// </summary>
    /// <param name="pattern">Pattern.</param>
    /// <returns>Match result.</returns>
    object this[IPattern pattern] { get; }

    /// <summary>
    /// Get match result by pattern.
    /// </summary>
    /// <typeparam name="T">Match result type.</typeparam>
    /// <param name="pattern">Pattern.</param>
    /// <returns>Match result.</returns>
    public T Get<T>(IPattern<T> pattern) => (T)this[pattern];

    /// <summary>
    /// Get two match results by pattern.
    /// </summary>
    /// <typeparam name="T1">Match result type 1.</typeparam>
    /// <typeparam name="T2">Match result type 2.</typeparam>
    /// <param name="pattern1">Pattern 1.</param>
    /// <param name="pattern2">Pattern 2.</param>
    /// <returns>Match results.</returns>
    public (T1 Result1, T2 Result2) Get<T1, T2>(IPattern<T1> pattern1, IPattern<T2> pattern2)
        => (Get(pattern1), Get(pattern2));

    /// <summary>
    /// Get two match results by pattern.
    /// </summary>
    /// <typeparam name="T1">Match result type 1.</typeparam>
    /// <typeparam name="T2">Match result type 2.</typeparam>
    /// <typeparam name="T3">Match result type 3.</typeparam>
    /// <param name="pattern1">Pattern 1.</param>
    /// <param name="pattern2">Pattern 2.</param>
    /// <param name="pattern3">Pattern 3.</param>
    /// <returns>Match results.</returns>
    public (T1 Result1, T2 Result2, T3 Result3) Get<T1, T2, T3>(IPattern<T1> pattern1, IPattern<T2> pattern2, IPattern<T3> pattern3)
        => (Get(pattern1), Get(pattern2), Get(pattern3));
}
