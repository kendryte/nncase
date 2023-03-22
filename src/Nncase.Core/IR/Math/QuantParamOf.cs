// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
using System;
using Nncase.PatternMatch;
using static Nncase.IR.TypePatternUtility;

namespace Nncase.IR.Math;

/// <summary>
/// QuantParamOf expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class QuantParamOf : Op
{
    /// <summary>
    /// Gets range.
    /// </summary>
    public static readonly ParameterInfo Range = new(typeof(QuantParamOf), 0, "range", HasShape(new Shape(2)));

    /// <summary>
    /// Gets bits.
    /// </summary>
    public static readonly ParameterInfo Bits = new(typeof(QuantParamOf), 1, "bits", IsIntegralScalar());

    public QuantMode QuantMode { get; }

    /// <inheritdoc/>
    public override string DisplayProperty() => $"QuantMode.{QuantMode}";
}
