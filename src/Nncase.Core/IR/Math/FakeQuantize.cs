// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
using System;
using Nncase.PatternMatch;

namespace Nncase.IR.Math;

/// <summary>
/// Fake quantize expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class FakeQuantize : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(FakeQuantize), 0, "input");

    /// <summary>
    /// Gets QuantParam.
    /// </summary>
    public static readonly ParameterInfo QuantParam = new(typeof(FakeQuantize), 1, "quantParam");

    public DataType TargetType { get; }

    /// <inheritdoc/>
    public override string DisplayProperty() => $"{TargetType.GetCSharpName()}";
}
