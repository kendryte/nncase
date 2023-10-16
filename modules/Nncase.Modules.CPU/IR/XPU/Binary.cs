﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

namespace Nncase.IR.XPU;

public sealed partial class Binary : XPUKernelOp
{
    public static readonly ParameterInfo Lhs = new(typeof(Binary), 0, "input");

    public static readonly ParameterInfo Rhs = new(typeof(Binary), 1, "input");

    public static readonly ParameterInfo Output = new(typeof(Binary), 2, "output");

    public BinaryOp BinaryOp { get; }

    /// <inheritdoc/>
    public override string DisplayProperty()
    {
        return $"BinaryOp.{BinaryOp}";
    }
}