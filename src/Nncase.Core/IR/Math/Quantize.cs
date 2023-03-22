// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
using System;
using Nncase.PatternMatch;
using static Nncase.IR.TypePatternUtility;

namespace Nncase.IR.Math;

/// <summary>
/// Quantize expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class Quantize : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(Quantize), 0, "input");

    /// <summary>
    /// Gets QuantParam.
    /// </summary>
    public static readonly ParameterInfo QuantParam = new(typeof(Quantize), 1, "quantParam", IsQuantParamType());

    public DataType TargetType { get; }

    /// <inheritdoc/>
    public override string DisplayProperty() => $"{TargetType.GetCSharpName()}";
}
