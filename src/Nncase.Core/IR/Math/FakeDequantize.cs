// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
using System;
using Nncase.PatternMatch;

namespace Nncase.IR.Math;

/// <summary>
/// Fake dequantize expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class FakeDequantize : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(FakeDequantize), 0, "input");

    /// <summary>
    /// Gets DequantParam.
    /// </summary>
    public static readonly ParameterInfo DequantParam = new(typeof(FakeDequantize), 1, "dequantParam");

    public DataType TargetType { get; }

    /// <inheritdoc/>
    public override string DisplayProperty() => $"{TargetType.GetCSharpName()}";
}
