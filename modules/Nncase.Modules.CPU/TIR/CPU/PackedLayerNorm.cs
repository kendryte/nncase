// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.PatternMatch;

namespace Nncase.TIR.CPU;

public sealed partial class PackedLayerNorm : CPUKernelOp
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

    public static readonly ParameterInfo Output = new(typeof(PackedLayerNorm), 3, "output", ParameterKind.Input);

    public int Axis { get; }

    public float Epsilon { get; }

    public bool UseMean { get; }

    public IRArray<int> PackedAxes { get; }

    public IRArray<int> PadedNums { get; }

    public DistributedType DistType { get; }

    public override string DisplayProperty() => $"Axis: {Axis}, Epsilon: {Epsilon}, UseMean: {UseMean}, PackedAxes: {PackedAxes}, PadedNums: {PadedNums}";
}
