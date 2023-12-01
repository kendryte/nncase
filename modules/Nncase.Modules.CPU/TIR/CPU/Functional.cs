// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.TIR;
using Nncase.TIR.CPU;

namespace Nncase.TIR.F;

public partial class CPU
{
    public static Call TensorLoad(Expr dest, Expr src, IRArray<SBP> ndsbp, Placement placement)
    {
        return new Call(new TensorLoad(ndsbp, placement), dest, src);
    }

    public static Call TensorStore(Expr src, Expr dest, IRArray<SBP> ndsbp, Placement placement)
    {
        return new Call(new TensorStore(ndsbp, placement), src, dest);
    }

    public static Call Unary(UnaryOp unaryOp, Expr input, Expr output)
    {
        return new Call(new Unary(unaryOp), input, output);
    }
}
