// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
using Nncase.IR;

namespace Nncase.TIR.NTT;

public sealed partial class ShapeOf : NTTKernelOp
{
    public static readonly ParameterInfo Input = new(typeof(ShapeOf), 0, "input");

    public static readonly ParameterInfo Output = new(typeof(ShapeOf), 1, "output");
}
