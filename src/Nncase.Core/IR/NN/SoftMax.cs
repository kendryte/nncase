﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.PatternMatch;
using static Nncase.IR.TypePatternUtility;

namespace Nncase.IR.NN;

/// <summary>
/// LogSoftmax expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed record LogSoftmax() : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(LogSoftmax), 0, "input");

    /// <summary>
    /// Gets axis.
    /// </summary>
    public static readonly ParameterInfo Axis = new(typeof(LogSoftmax), 1, "axis", IsIntegralScalar());
}

/// <summary>
/// Softmax expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed record Softmax() : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(Softmax), 0, "input");

    /// <summary>
    /// Gets axis.
    /// </summary>
    public static readonly ParameterInfo Axis = new(typeof(Softmax), 1, "axis", IsIntegralScalar());
}

/// <summary>
/// Softplus expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed record Softplus() : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(Softplus), 0, "input");
}

/// <summary>
/// Softsign expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed record Softsign() : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(Softsign), 0, "input");
}
