// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.Passes.Rules.CPU;

public static class PackRule
{
    // public static Expr PackedKernel(Expr input, int axis, Func<Expr, Expr> kernel, CompileSession session)
    // {
    //     int lanes = 256 / 8 / input.CheckedDataType.SizeInBytes;
    //     var packedInput = IR.F.CPU.Pack(input, lanes, axis);
    //     var packedOutput = kernel(packedInput);
    //     return IR.F.CPU.Unpack(packedOutput, IR.F.Tensors.ShapeOf(input)[axis], axis);
    // }
}
