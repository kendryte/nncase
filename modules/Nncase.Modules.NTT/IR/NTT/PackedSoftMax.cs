// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.PatternMatch;

namespace Nncase.IR.NTT;

[PatternFunctionalGenerator]
public sealed partial class VectorizedSoftmax : Op
{
    public static readonly ParameterInfo Input = new(typeof(VectorizedSoftmax), 0, "input", ParameterKind.Input);

    public int Axis { get; }

    public IRArray<int> VectorizedAxes { get; }

    public override string DisplayProperty() => $"{Axis}, {VectorizedAxes}";
}
