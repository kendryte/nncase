// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.TIR.CPU;

public sealed partial class ScatterND : CPUKernelOp
{
    public static readonly ParameterInfo Input = new(typeof(ScatterND), 0, "input");

    public static readonly ParameterInfo Indices = new(typeof(ScatterND), 1, "indices");

    public static readonly ParameterInfo Updates = new(typeof(ScatterND), 2, "updates");

    public static readonly ParameterInfo Output = new(typeof(ScatterND), 3, "output");
}
