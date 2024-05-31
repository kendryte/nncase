// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.PatternMatch;
using static Nncase.IR.TypePatternUtility;

namespace Nncase.TIR.CPU;

public sealed partial class PackedTranspose : CPUKernelOp
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(PackedTranspose), 0, "input", ParameterKind.Input);

    public static readonly ParameterInfo Output = new(typeof(PackedTranspose), 1, "output", ParameterKind.Input);

    public IRArray<int> Perm { get; }

    public IRArray<int> PackedAxes { get; }
}
