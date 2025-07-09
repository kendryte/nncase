// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.PatternMatch;
using static Nncase.IR.TypePatternUtility;

namespace Nncase.IR.NTT;

[PatternFunctionalGenerator]
public sealed partial class PackedLayerNorm : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(PackedLayerNorm), 0, "input", ParameterKind.Input);

    /// <summary>
    /// Gets scale.
    /// </summary>
    public static readonly ParameterInfo Scale = new(typeof(PackedLayerNorm), 1, "scale", ParameterKind.Input);

    /// <summary>
    /// Gets bias.
    /// </summary>
    public static readonly ParameterInfo Bias = new(typeof(PackedLayerNorm), 2, "bias", ParameterKind.Input);

    public static readonly ParameterInfo PadedNums = new(typeof(PackedLayerNorm), 3, "padedNums", IsShapeType());

    public int Axis { get; }

    public float Epsilon { get; }

    public bool UseMean { get; }

    public IRArray<int> PackedAxes { get; }

    public override string DisplayProperty() => $"Axis: {Axis}, Epsilon: {Epsilon}, UseMean: {UseMean}, PackedAxes: {PackedAxes}";
}
