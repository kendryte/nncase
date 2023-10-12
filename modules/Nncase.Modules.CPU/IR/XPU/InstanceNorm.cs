// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

namespace Nncase.IR.XPU;

public sealed partial class InstanceNorm : XPUKernelOp
{
    public static readonly ParameterInfo Input = new(typeof(InstanceNorm), 0, "input");

    public static readonly ParameterInfo Scale = new(typeof(InstanceNorm), 1, "scale");

    public static readonly ParameterInfo Bias = new(typeof(InstanceNorm), 2, "bias");

    public static readonly ParameterInfo Output = new(typeof(InstanceNorm), 3, "output");

    public float Epsilon { get; }

    public DistributedType DistType { get; }
}
