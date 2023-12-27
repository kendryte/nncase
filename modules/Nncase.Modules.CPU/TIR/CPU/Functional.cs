// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.TIR;
using Nncase.TIR.CPU;

namespace Nncase.TIR.F;

public partial class CPU
{
    public static Call PtrOf(string name, DataType primType) => new Call(new PtrOf(name, primType));

    public static Call SramPtr(Expr input, DataType primType) => new Call(new SramPtr(primType), input);

    public static Call TensorLoad(Expr dest, Expr src, IRArray<SBP> ndsbp, Placement placement)
    {
        return new Call(new TensorLoad(ndsbp, placement), dest, src);
    }

    public static Call TensorStore(Expr src, Expr dest, IRArray<SBP> ndsbp, Placement placement)
    {
        return new Call(new TensorStore(ndsbp, placement), src, dest);
    }

    public static Call Memcopy(Expr dest, Expr src)
    {
        return new Call(new Memcopy(), dest, src);
    }

    public static Call Unary(UnaryOp unaryOp, Expr input, Expr output)
    {
        return new Call(new TIR.CPU.Unary(unaryOp), input, output);
    }

    public static Call Binary(BinaryOp binaryOp, DistributedType ltype, DistributedType rtype, DistributedType outtype, Expr lhs, Expr rhs, Expr output)
    {
        return new Call(new TIR.CPU.Binary(binaryOp, ltype, rtype, outtype), lhs, rhs, output);
    }

    public static Call Matmul(Expr lhs, Expr rhs, Expr output)
    {
        return new Call(new Matmul(), lhs, rhs, output);
    }
}
