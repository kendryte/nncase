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

    public static Call Binary(BinaryOp binaryOp, Expr lhs, Expr rhs, Expr output)
    {
        return new Call(new TIR.CPU.Binary(binaryOp), lhs, rhs, output);
    }

    public static Call Matmul(Expr lhs, Expr rhs, Expr output)
    {
        return new Call(new Matmul(), lhs, rhs, output);
    }

    public static Expr Pack(Expr input, Expr output, IRArray<int> lanes, IRArray<int> axes)
    {
        return new Call(new Pack(lanes, axes), input, output);
    }

    public static Expr Unpack(Expr input, Expr output, IRArray<int> axes)
    {
        return new Call(new Unpack(axes), input, output);
    }

    public static Expr PackedSoftmax(Expr input, Expr output, int axis, IRArray<int> packedAxes)
    {
        return new Call(new PackedSoftmax(axis, packedAxes), input, output);
    }

    public static Expr PackedLayerNorm(Expr input, Expr scale, Expr bias, Expr output, int axis, float epsilon, bool usemean, IRArray<int> packedAxes, IRArray<int> padedNums)
    {
        return new Call(new PackedLayerNorm(axis, epsilon, usemean, packedAxes, padedNums), input, scale, bias, output);
    }

    public static Expr PackedMatMul(Expr lhs, Expr rhs, Expr output, IRArray<int> lhsPackedAxes, IRArray<int> lhsPadedNums, IRArray<int> rhsPackedAxes, IRArray<int> rhsPadedNums)
    {
        return new Call(new PackedMatMul(lhsPackedAxes, lhsPadedNums, rhsPackedAxes, rhsPadedNums), lhs, rhs, output);
    }

    public static Expr PackedBinary(Expr lhs, Expr rhs, Expr output, BinaryOp binaryOp, IRArray<int> lhsPackedAxes, IRArray<int> lhsPadedNums, IRArray<int> rhsPackedAxes, IRArray<int> rhsPadedNums)
    {
        return new Call(new PackedBinary(binaryOp, lhsPackedAxes, lhsPadedNums, rhsPackedAxes, rhsPadedNums), lhs, rhs, output);
    }

    public static Expr PackedTranspose(Expr input, Expr perm, Expr output, IRArray<int> packedAxes)
    {
        return new Call(new PackedTranspose(packedAxes), input, perm, output);
    }
}
