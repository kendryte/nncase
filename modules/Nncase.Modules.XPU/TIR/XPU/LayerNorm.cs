// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
using Nncase.IR;

namespace Nncase.TIR.XPU;

public sealed partial class LayerNorm : XPUKernelOp
{
    public static readonly ParameterInfo Input = new(typeof(LayerNorm), 0, "input");

    public static readonly ParameterInfo Scale = new(typeof(LayerNorm), 1, "scale");

    public static readonly ParameterInfo Bias = new(typeof(LayerNorm), 2, "bias");

    public static readonly ParameterInfo Output = new(typeof(LayerNorm), 3, "output");

    public int Axis { get; }

    public float Epsilon { get; }

    public bool UseMean { get; }

    public DistributedType DistType { get; }
}
