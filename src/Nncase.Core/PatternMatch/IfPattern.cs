// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.IR;
using Nncase.IR.Tensors;

namespace Nncase.PatternMatch;

/// <summary>
/// Pattern for <see cref="If"/>.
/// </summary>
/// <param name="Then">Then pattern.</param>
/// <param name="Else">Else pattern.</param>
/// <param name="Arguments">Arguments pattern.</param>
/// <param name="Name"> name. </param>
public sealed record IfPattern(Pattern Then, Pattern Else, VArgsPattern Arguments, string? Name) : Pattern<If>(Name)
{
    /// <summary>
    /// Initializes a new instance of the <see cref="IfPattern"/> class.
    /// </summary>
    /// <param name="if"><see cref="If"/> expression.</param>
    /// <param name="name">name.</param>
    public IfPattern(If @if, string? name)
        : this(@if.Then, @if.Else, new VArgsPattern(@if.Arguments.ToArray(), null), name)
    {
    }

    /// <summary>
    /// Get parameter pattern.
    /// </summary>
    /// <param name="parameter">Parameter info.</param>
    /// <returns>Parameter pattern.</returns>
    public Pattern this[ParameterInfo parameter]
    {
        get => Arguments.Fields[parameter.Index];
    }
}

/// <summary>
/// PatternMatch Utility.
/// </summary>
public static partial class Utility
{
    /// <summary>
    /// is call .
    /// </summary>
    /// <param name="name">name.</param>
    /// <param name="then">then.</param>
    /// <param name="else">else.</param>
    /// <param name="parameters">params.</param>
    /// <returns>call pattern.</returns>
    public static IfPattern IsIf(string? name, Pattern then, Pattern @else, VArgsPattern parameters) => new IfPattern(then, @else, parameters, name);

    /// <summary>
    /// is call .
    /// </summary>
    /// <param name="then">then.</param>
    /// <param name="else">else.</param>
    /// <param name="parameters">params.</param>
    /// <returns>call pattern.</returns>
    public static IfPattern IsIf(Pattern then, Pattern @else, VArgsPattern parameters) => IsIf(null, then, @else, parameters);

    /// <summary>
    /// is call .
    /// </summary>
    /// <param name="name">name.</param>
    /// <param name="then">then.</param>
    /// <param name="else">else.</param>
    /// <param name="parameters">params.</param>
    /// <returns>call pattern.</returns>
    public static IfPattern IsIf(string? name, Pattern then, Pattern @else, params Pattern[] parameters) => new IfPattern(then, @else, new VArgsPattern(parameters, null), name);

    /// <summary>
    /// is call .
    /// </summary>
    /// <param name="name">name.</param>
    /// <param name="then">then.</param>
    /// <param name="else">else.</param>
    /// <param name="parameters">params.</param>
    public static IfPattern IsIf(string? name, FunctionPattern then, FunctionPattern @else, params Pattern[] parameters) => new IfPattern(then, @else, new VArgsPattern(parameters, null), name);

    /// <summary>
    /// is call .
    /// </summary>
    /// <param name="name">name.</param>
    /// <param name="then">then.</param>
    /// <param name="else">else.</param>
    /// <param name="parameters">params.</param>
    public static IfPattern IsIf(string? name, FunctionPattern then, FunctionPattern @else, VArgsPattern parameters) => new IfPattern(then, @else, parameters, name);

    /// <summary>
    /// is call .
    /// </summary>
    /// <param name="then">then.</param>
    /// <param name="else">else.</param>
    /// <param name="parameters">params.</param>
    public static IfPattern IsIf(Pattern then, Pattern @else, params Pattern[] parameters) => IsIf(null, then, @else, parameters);
}
