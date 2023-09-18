// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR.XPU;

namespace Nncase.IR.F;

public partial class XPU
{
    public static Call TDMALoad(Expr dest, Expr src, IRArray<SBP> ndsbp, Placement placement)
    {
        return new Call(new TDMALoad(ndsbp, placement), dest, src);
    }

    public static Call TDMAStore(Expr src, Expr dest, IRArray<SBP> ndsbp, Placement placement)
    {
        return new Call(new TDMAStore(ndsbp, placement), src, dest);
    }

    public static Call Unary(UnaryOp unaryOp, Expr input, Expr output)
    {
        return new Call(new Unary(unaryOp), input, output);
    }

    public static Call Binary(BinaryOp binaryOp, Expr lhs, Expr rhs, Expr output)
    {
        return new Call(new Binary(binaryOp), lhs, rhs, output);
    }

    public static Call Matmul(Expr lhs, Expr rhs, Expr output)
    {
        return new Call(new Matmul(), lhs, rhs, output);
    }

    public static Call LayerNorm(int axis, float eps, bool useMean, Expr input, Expr scale, Expr bias, Expr output, DistributedType distributedType)
    {
        return new Call(new LayerNorm(axis, eps, useMean, distributedType), input, scale, bias, output);
    }

    public static Call Gather(int axis, Expr input, Expr indices, Expr output)
    {
        return new Call(new IR.XPU.Gather(axis), input, indices, output);
    }

    public static Call Concat(int axis, Expr[] inputs, Expr output)
    {
        return new Call(new Concat(axis), inputs.Concat(new[] { output }).ToArray());
    }

    internal static Expr Slice(TIR.Buffer input, TIR.Buffer output, Expr begins, Expr ends, Expr axes, DistributedType distributedType) => new Call(new Slice(distributedType), input, output, begins, ends, axes);
}
