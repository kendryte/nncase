// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
using System;

namespace Nncase.IR.Math;

/// <summary>
/// Dequantize expression.
/// </summary>
public sealed record Dequantize(DataType TargetType) : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(Dequantize), 0, "input");

    /// <summary>
    /// Gets zero-point.
    /// </summary>
    public static readonly ParameterInfo ZeroPoint = new(typeof(Dequantize), 1, "zero_point");

    /// <summary>
    /// Gets scale.
    /// </summary>
    public static readonly ParameterInfo Scale = new(typeof(Dequantize), 2, "scale");
}
