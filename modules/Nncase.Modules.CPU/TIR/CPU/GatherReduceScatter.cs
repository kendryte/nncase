// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
using Nncase.IR;

namespace Nncase.TIR.CPU;

public sealed partial class GatherReduceScatter : CPUKernelOp
{
    public static readonly ParameterInfo Input = new(typeof(GatherReduceScatter), 0, "input");

    public static readonly ParameterInfo Output = new(typeof(GatherReduceScatter), 1, "output");

    public DistributedType InType { get; }

    public DistributedType OutType { get; }
}
