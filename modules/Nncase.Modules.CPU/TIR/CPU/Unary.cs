﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
using Nncase.IR;

namespace Nncase.TIR.CPU;

public sealed partial class Unary : CPUKernelOp
{
    public static readonly ParameterInfo Input = new(typeof(Unary), 0, "input");

    public static readonly ParameterInfo Output = new(typeof(Unary), 1, "output");

    public UnaryOp UnaryOp { get; }

    /// <inheritdoc/>
    public override string DisplayProperty()
    {
        return $"UnaryOp.{UnaryOp}";
    }
}
