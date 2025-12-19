// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.NN;

namespace Nncase.TIR.NTT;

public sealed partial class TopK : Nncase.TIR.NTT.NTTKernelOp
{
    public static readonly ParameterInfo X = new(typeof(TopK), 0, "X", ParameterKind.Input);
    public static readonly ParameterInfo K = new(typeof(TopK), 1, "K", ParameterKind.Input);

    // public static readonly ParameterInfo Axis = new(typeof(TopK), 2, "Axis", ParameterKind.Input);
    public long Axis { get; }

    public long Largest { get; }

    public long Sorted { get; }
}
