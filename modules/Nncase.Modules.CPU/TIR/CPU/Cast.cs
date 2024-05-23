// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
using Nncase.IR;

namespace Nncase.TIR.CPU;

public sealed partial class Cast : CPUKernelOp
{
    public static readonly ParameterInfo Input = new(typeof(Cast), 0, "input");

    public static readonly ParameterInfo Output = new(typeof(Cast), 1, "output");

    public DataType NewType { get; }

    public CastMode CastMode { get; }

    /// <inheritdoc/>
    public override string DisplayProperty() => $"{NewType.GetCSharpName()}, CastMode.{CastMode}";
}
