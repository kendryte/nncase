// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.IR;
using Nncase.IR.Tensors;

namespace Nncase.PatternMatch;

/// <summary>
/// Pattern for <see cref="Call"/>.
/// </summary>
/// <param name="Target">Target pattern.</param>
/// <param name="Arguments">Arguments pattern.</param>
/// <param name="Name"> name. </param>
public sealed record CallPattern(Pattern Target, VArgsPattern Arguments, string? Name) : Pattern<Call>(Name)
{
    /// <summary>
    /// Initializes a new instance of the <see cref="CallPattern"/> class.
    /// </summary>
    /// <param name="call"><see cref="Call"/> expression.</param>
    /// <param name="name">name.</param>
    public CallPattern(Call call, string? name)
        : this(call.Target, new VArgsPattern(call.Arguments.ToArray(), null), name)
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
    /// <param name="target">target.</param>
    /// <param name="parameters">params.</param>
    /// <returns>call pattern.</returns>
    public static CallPattern IsCall(string? name, Pattern target, VArgsPattern parameters) => new CallPattern(target, parameters, name);

    /// <summary>
    /// is call .
    /// </summary>
    /// <param name="target">target.</param>
    /// <param name="parameters">params.</param>
    /// <returns>call pattern.</returns>
    public static CallPattern IsCall(Pattern target, VArgsPattern parameters) => IsCall(null, target, parameters);

    /// <summary>
    /// is call .
    /// </summary>
    /// <param name="name">name.</param>
    /// <param name="target">target.</param>
    /// <param name="parameters">params.</param>
    /// <returns>call pattern.</returns>
    public static CallPattern IsCall(string? name, Pattern target, params Pattern[] parameters) => new CallPattern(target, new VArgsPattern(parameters, null), name);

    /// <summary>
    /// is call .
    /// </summary>
    /// <param name="name">name.</param>
    /// <param name="target">function target.</param>
    /// <param name="parameters">params.</param>
    public static CallPattern IsCall(string? name, FunctionPattern target, params Pattern[] parameters) => new CallPattern(target, new VArgsPattern(parameters, null), name);

    /// <summary>
    /// is call .
    /// </summary>
    /// <param name="name">name.</param>
    /// <param name="target">function target.</param>
    /// <param name="parameters">params.</param>
    public static CallPattern IsCall(string? name, FunctionPattern target, VArgsPattern parameters) => new CallPattern(target, parameters, name);

    /// <summary>
    /// is call .
    /// </summary>
    /// <param name="target">function target.</param>
    /// <param name="parameters">params.</param>
    public static CallPattern IsCall(Pattern target, params Pattern[] parameters) => IsCall(null, target, parameters);

    /// <summary>
    /// is call .
    /// </summary>
    /// <param name="name">name.</param>
    /// <param name="target">op target.</param>
    /// <param name="parameters">params.</param>
    public static CallPattern IsCall<T>(string? name, OpPattern<T> target, params Pattern[] parameters)
        where T : Op
        => new CallPattern(target, new VArgsPattern(parameters, null), name);

    /// <summary>
    /// is call .
    /// </summary>
    /// <param name="name">name.</param>
    /// <param name="target">op target.</param>
    /// <param name="parameters">params.</param>
    public static CallPattern IsCall<T>(string? name, OpPattern<T> target, VArgsPattern parameters)
        where T : Op
        => new CallPattern(target, parameters, name);
}
