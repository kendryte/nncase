// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
using Nncase.IR;

namespace Nncase.TIR.XPU;

public sealed partial class Conv2D : XPUKernelOp
{
    public static readonly ParameterInfo Input = new(typeof(Conv2D), 0, "input");

    public static readonly ParameterInfo Weights = new(typeof(Conv2D), 1, "weights");

    public static readonly ParameterInfo Bias = new(typeof(Conv2D), 2, "bias");

    public IRArray<int> Stride { get; }

    public IRArray<int> Padding { get; }

    public IRArray<int> Dilation { get; }

    public int Groups { get; }

    public TensorConst FusedClamp { get; }

    public DistributedType DistType { get; }
}
