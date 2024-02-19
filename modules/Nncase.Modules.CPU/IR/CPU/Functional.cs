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

    public static Call Load(Expr input)
    {
        return new Call(new Load(), input);
    }

    public static Call Store(Expr input)
    {
        return new Call(new Store(), input);
    }

    public static Call Pack(Expr input, int[] lanes, int[] axes)
    {
        return new Call(new Pack(lanes, axes), input);
    }

    public static Call Unpack(Expr input, int[] axes)
    {
        return new Call(new Unpack(axes), input);
    }

    public static Expr PackedSoftMax(Expr input, int axis, IRArray<int> packedAxes)
    {
        return new Call(new PackedSoftMax(axis, packedAxes), input);
    }
}
