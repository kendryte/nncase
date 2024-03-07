// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.PatternMatch;

namespace Nncase.TIR.CPU;

public sealed partial class PackedSoftmax : CPUKernelOp
{
    public static readonly ParameterInfo Input = new(typeof(PackedSoftmax), 0, "input", ParameterKind.Input);

    public static readonly ParameterInfo Output = new(typeof(PackedSoftmax), 1, "output", ParameterKind.Input);

    public int Axis { get; }

    public IRArray<int> PackedAxes { get; }

    public override string DisplayProperty() => $"{Axis}, {PackedAxes}";
}
