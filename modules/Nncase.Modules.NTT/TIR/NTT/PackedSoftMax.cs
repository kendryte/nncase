// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.PatternMatch;

namespace Nncase.TIR.NTT;

public sealed partial class VectorizedSoftmax : NTTKernelOp
{
    public static readonly ParameterInfo Input = new(typeof(VectorizedSoftmax), 0, "input", ParameterKind.Input);

    public static readonly ParameterInfo Output = new(typeof(VectorizedSoftmax), 1, "output", ParameterKind.Input);

    public int Axis { get; }

    public IRArray<int> VectorizedAxes { get; }

    public override string DisplayProperty() => $"{Axis}, {VectorizedAxes}";
}
