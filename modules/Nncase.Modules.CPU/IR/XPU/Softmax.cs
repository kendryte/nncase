﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

namespace Nncase.IR.XPU;

public sealed partial class Softmax : XPUKernelOp
{
    public static readonly ParameterInfo Input = new(typeof(Softmax), 0, "input");

    public static readonly ParameterInfo Output = new(typeof(Softmax), 1, "output");

    public int Axis { get; }

    public DistributedType DistType { get; }
}