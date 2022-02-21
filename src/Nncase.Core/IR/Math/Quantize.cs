// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
using Nncase.PatternMatch;
using System;

namespace Nncase.IR.Math;

/// <summary>
/// Quantize expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed record Quantize(DataType TargetType) : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(Quantize), 0, "input");

    /// <summary>
    /// Gets zero-point.
    /// </summary>
    public static readonly ParameterInfo ZeroPoint = new(typeof(Quantize), 1, "zeroPoint");

    /// <summary>
    /// Gets scale.
    /// </summary>
    public static readonly ParameterInfo Scale = new(typeof(Quantize), 2, "scale");
}
