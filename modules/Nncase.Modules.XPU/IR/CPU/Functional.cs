// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR.CPU;

namespace Nncase.IR.F;

public partial class CPU
{
    /// <summary>
    /// Call cpu kernel.
    /// </summary>
    /// <param name="target">Unary operator.</param>
    /// <param name="inputs">Source inputs.</param>
    /// <returns>Result expression.</returns>
    public static Call CPUKernel(Op target, params Expr[] inputs)
    {
        return new Call(new CPUKernelOp(target), inputs);
    }

    public static Call Boxing(Expr input, IRType type)
    {
        return new Call(new Boxing(type), input);
    }
}
