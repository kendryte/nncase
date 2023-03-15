// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.IR;

namespace Nncase.PatternMatch;

/// <summary>
/// Pattern for <see cref="Function"/>.
/// </summary>
/// <param name="Body">Body pattern.</param>
/// <param name="Parameters">Parameters pattern.</param>
/// <param name="Name">Name.</param>
public sealed record FunctionPattern(Pattern Body, VArgsPattern Parameters, string? Name) : Pattern<Function>(Name)
{
    /// <summary>
    /// Initializes a new instance of the <see cref="FunctionPattern"/> class.
    /// </summary>
    /// <param name="function"><see cref="Function"/> expression.</param>
    /// <param name="name">name.</param>
    public FunctionPattern(Function function, string? name)
        : this(function.Body, new VArgsPattern(function.Parameters.AsValueEnumerable().Select(x => (Pattern)x).ToArray(), null), name)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="FunctionPattern"/> class.
    /// </summary>
    /// <param name="body">Body pattern.</param>
    /// <param name="parameters">Parameter patterns.</param>
    /// <param name="name">name.</param>
    public FunctionPattern(Pattern body, Pattern[] parameters, string? name)
        : this(body, new VArgsPattern(parameters, null), name)
    {
    }
}

public static partial class Utility
{
    /// <summary>
    /// Create the function pattern.
    /// </summary>
    /// <param name="name">name.</param>
    /// <param name="body">body.</param>
    /// <param name="parameters">params.</param>
    /// <returns>FunctionPattern .</returns>
    public static FunctionPattern IsFunction(string? name, Pattern body, VArgsPattern parameters) => new FunctionPattern(body, parameters, name);

    public static FunctionPattern IsFunction(Pattern body, VArgsPattern parameters) => IsFunction(null, body, parameters);

    public static FunctionPattern IsFunction(string? name, Pattern body, params Pattern[] parameters) => IsFunction(name, body, IsVArgs(parameters));

    public static FunctionPattern IsFunction(Pattern body, params Pattern[] parameters) => IsFunction(null, body, IsVArgs(parameters));
}
