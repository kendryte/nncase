// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.PatternMatch;

namespace Nncase.IR.CPU;

[PatternFunctionalGenerator]
public sealed partial class InstacneNorm : PackedOp
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(InstacneNorm), 0, "input", ParameterKind.Input);

    /// <summary>
    /// Gets scale.
    /// </summary>
    public static readonly ParameterInfo Scale = new(typeof(InstacneNorm), 1, "scale", ParameterKind.Input);

    /// <summary>
    /// Gets bias.
    /// </summary>
    public static readonly ParameterInfo Bias = new(typeof(InstacneNorm), 2, "bias", ParameterKind.Input);

    public float Epsilon { get; }

    public IRArray<int> PackedAxes { get; }

    public IRArray<int> PadedNums { get; }

    public override string DisplayProperty() => $"Epsilon: {Epsilon}, PackedAxes: {PackedAxes}, PadedNums: {PadedNums}";
}
