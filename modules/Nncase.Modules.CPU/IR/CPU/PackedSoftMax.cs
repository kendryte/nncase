// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.PatternMatch;

namespace Nncase.IR.CPU;

[PatternFunctionalGenerator]
public sealed partial class PackedSoftmax : PackedOp
{
    public static readonly ParameterInfo Input = new(typeof(PackedSoftmax), 0, "input", ParameterKind.Input);

    public int Axis { get; }

    public IRArray<int> PackedAxes { get; }

    public override string DisplayProperty() => $"{Axis}, {PackedAxes}";
}
