// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
using Nncase.IR;

namespace Nncase.TIR.NTT;

public sealed partial class Cast : NTTKernelOp
{
    public static readonly ParameterInfo Input = new(typeof(Cast), 0, "input");

    public static readonly ParameterInfo Output = new(typeof(Cast), 1, "output");

    public DataType NewType { get; }

    public CastMode CastMode { get; }

    public IRArray<int> VectorizeAxes { get; }

    /// <inheritdoc/>
    public override string DisplayProperty() => $"{NewType.GetCSharpName()}, CastMode.{CastMode}, VectorizeAxes: {string.Join(",", VectorizeAxes.IsDefaultOrEmpty ? Array.Empty<int>() : VectorizeAxes.ToArray())}";
}
