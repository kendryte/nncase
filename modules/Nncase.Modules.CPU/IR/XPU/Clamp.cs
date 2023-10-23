// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Dynamic;

namespace Nncase.IR.XPU;

public sealed partial class Clamp : XPUKernelOp
{
    public static readonly ParameterInfo Input = new(typeof(Clamp), 0, "input");

    public static readonly ParameterInfo Output = new(typeof(Clamp), 1, "output");

    public float Min { get; }

    public float Max { get; }
}
