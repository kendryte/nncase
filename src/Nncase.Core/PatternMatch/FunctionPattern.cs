// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Nncase.IR;

namespace Nncase.PatternMatch;

/// <summary>
/// Pattern for <see cref="Function"/>.
/// </summary>
/// <param name="Body">Body pattern.</param>
/// <param name="Parameters">Parameters pattern.</param>
public sealed record FunctionPattern(Pattern Body, VArgsPattern Parameters) : Pattern<Function>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="FunctionPattern"/> class.
    /// </summary>
    /// <param name="function"><see cref="Function"/> expression.</param>
    public FunctionPattern(Function function)
        : this(function.Body, new VArgsPattern(function.Parameters.Select(x => (Pattern)x).ToArray()))
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="FunctionPattern"/> class.
    /// </summary>
    /// <param name="body">Body pattern.</param>
    /// <param name="parameters">Parameter patterns.</param>
    public FunctionPattern(Pattern body, params ExprPattern[] parameters)
        : this(body, new VArgsPattern(parameters))
    {
    }
}

public static partial class Utility
{
    public static FunctionPattern IsFunction(ExprPattern Body, VArgsPattern Parameters) => new FunctionPattern(Body, Parameters);
}
