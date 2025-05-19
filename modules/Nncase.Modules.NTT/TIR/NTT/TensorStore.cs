// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
using Nncase.IR;

namespace Nncase.TIR.NTT;

public sealed partial class TensorStore : NTTKernelOp
{
    public static readonly ParameterInfo Src = new(typeof(TensorStore), 0, "src");

    public static readonly ParameterInfo Dest = new(typeof(TensorStore), 1, "dest");

    public IRArray<SBP> NdSbp { get; }

    public Placement Placement { get; }
}
