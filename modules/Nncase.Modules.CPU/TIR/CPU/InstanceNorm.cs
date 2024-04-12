// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.PatternMatch;

namespace Nncase.TIR.CPU;

public sealed partial class InstanceNorm : CPUKernelOp
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(InstanceNorm), 0, "input", ParameterKind.Input);

    /// <summary>
    /// Gets scale.
    /// </summary>
    public static readonly ParameterInfo Scale = new(typeof(InstanceNorm), 1, "scale", ParameterKind.Input);

    /// <summary>
    /// Gets bias.
    /// </summary>
    public static readonly ParameterInfo Bias = new(typeof(InstanceNorm), 2, "bias", ParameterKind.Input);

    public static readonly ParameterInfo Output = new(typeof(InstanceNorm), 3, "output", ParameterKind.Input);

    public float Epsilon { get; }

    public IRArray<int> PackedAxes { get; }

    public IRArray<int> PadedNums { get; }

    public override string DisplayProperty() => $"Epsilon: {Epsilon}, PackedAxes: {PackedAxes}, PadedNums: {PadedNums}";
}
