// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
using Nncase.IR;

namespace Nncase.TIR.CPU;

public sealed partial class ConstantOfShape : CPUKernelOp
{
    public static readonly ParameterInfo Shape = new(typeof(ConstantOfShape), 0, "shape");
    public static readonly ParameterInfo Value = new(typeof(ConstantOfShape), 1, "value");
    public static readonly ParameterInfo Output = new(typeof(ConstantOfShape), 2, "output");

}
