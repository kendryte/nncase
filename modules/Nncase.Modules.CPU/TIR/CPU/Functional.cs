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
    /// <summary>
    /// the ptr of can create the *PtrName in the c code.
    /// </summary>
    /// <param name="name">c pointer name.</param>
    /// <param name="primType">type.</param>
    /// <returns>call.</returns>
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

    public static Call Matmul(Expr lhs, Expr rhs, Expr output, Expr loadC)
    {
        return new Call(new Matmul(), lhs, rhs, output, loadC);
    }

    public static Expr Pack(Expr input, Expr output, IRArray<int> lanes, IRArray<int> axes)
    {
        return new Call(new Pack(lanes, axes), input, output);
    }

    public static Call Conv2D(Buffer input, Buffer weights, Buffer bias, Buffer output, int[] stride, int[] padding, int[] dilation, int groups, PadMode padMode, DistributedType distributedType) => new Call(new Conv2D(stride, padding, dilation, groups, padMode, distributedType), input, weights, bias, output);

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
        return new Call(new PackedLayerNorm(axis, epsilon, usemean, packedAxes, padedNums, null!), input, scale, bias, output);
    }

    public static Expr InstanceNorm(Expr input, Expr scale, Expr bias, Expr output, float epsilon, IRArray<int> packedAxes, IRArray<int> padedNums, DistributedType distributedType)
    {
        return new Call(new InstanceNorm(epsilon, packedAxes, padedNums, distributedType), input, scale, bias, output);
    }

    public static Expr PackedMatMul(Expr lhs, Expr rhs, Expr output, IRArray<int> lhsPackedAxes, IRArray<int> lhsPadedNums, IRArray<int> rhsPackedAxes, IRArray<int> rhsPadedNums)
    {
        return new Call(new PackedMatMul(lhsPackedAxes, lhsPadedNums, rhsPackedAxes, rhsPadedNums), lhs, rhs, output);
    }

    public static Expr PackedBinary(Expr lhs, Expr rhs, Expr output, BinaryOp binaryOp, IRArray<int> lhsPackedAxes, IRArray<int> lhsPadedNums, IRArray<int> rhsPackedAxes, IRArray<int> rhsPadedNums)
    {
        return new Call(new PackedBinary(binaryOp, lhsPackedAxes, lhsPadedNums, rhsPackedAxes, rhsPadedNums), lhs, rhs, output);
    }

    public static Call ResizeImage(Buffer input, Buffer output, int[] packedAxes, int[] padedNums, int[] newSize, ImageResizeMode resizeMode, ImageResizeTransformationMode transformationMode, ImageResizeNearestMode nearestMode)
    {
        return new Call(new ResizeImage(packedAxes, padedNums, newSize, resizeMode, transformationMode, nearestMode), input, output);
    }

    public static Expr PackedTranspose(Expr input, Expr output, IRArray<int> perm, IRArray<int> packedAxes)
    {
        return new Call(new PackedTranspose(perm, packedAxes), input, output);
    }

    public static Expr Slice(Buffer input, Buffer ret, int[] begin, int[] stop, int[] axes, int[] stride, DistributedType distributedType)
    {
        return new Call(new Slice(begin, stop, axes, stride, distributedType), input, ret);
    }

    public static Expr Concat(Buffer[] inputs, Buffer ret, int axis)
    {
        return new Call(new Concat(axis), inputs.Concat(new[] { ret }).ToArray());
    }

    public static Expr Reshape(Buffer input, Buffer ret, int[] newShape)
    {
        return new Call(new Reshape(newShape), input, ret);
    }

    public static Expr Swish(Expr buffer, Expr ret, float v)
    {
        return new Call(new Swish(v), buffer, ret);
    }

    public static Expr Gather(Buffer input, Buffer indcies, Buffer ret, int axis)
    {
        return new Call(new Gather(axis), input, indcies, ret);
    }

    public static Expr Transpose(Buffer buffer, Buffer ret, int[] perm)
    {
        return new Call(new Transpose(perm), buffer, ret);
    }

    public static Expr Pad(Buffer input, Buffer ret, int[] pads, float padValue)
    {
        return new Call(new Pad(pads, padValue), input, ret);
    }

    public static Expr Im2col(Buffer input, Buffer output, IRArray<int> kernel, IRArray<int> stride, IRArray<int> padding, IRArray<int> packedAxes, IRArray<int> padedNums)
    {
        return new Call(new Im2col(kernel, stride, padding, packedAxes, padedNums), input, output);
    }

    public static Expr Reduce(Buffer input, Buffer ret, int[] packedAxes, int[] padedNums, IRArray<int> axis, bool keepDims, ReduceOp reduceOp)
    {
        return new Call(new TIR.CPU.Reduce(packedAxes, padedNums, axis, keepDims, reduceOp), input, ret);
    }

    public static Expr ReduceArg(Buffer input, Buffer ret, int axis, bool keepDims, bool selectLastIndex, ReduceArgOp reduceArgOp, DataType destType)
    {
        return new Call(new TIR.CPU.ReduceArg(axis, keepDims, selectLastIndex, reduceArgOp, destType), input, ret);
    }

    public static Call GatherReduceScatter(Expr input, Expr output, DistributedType inType, DistributedType outType)
    {
        return new Call(new TIR.CPU.GatherReduceScatter(inType, outType), input, output);
    }

    public static Call Clamp(Expr input, Expr output, float min, float max)
    {
        return new Call(new TIR.CPU.Clamp(min, max), input, output);
    }

    public static Call Cast(Expr input, Expr output, DataType newType, CastMode castMode)
    {
        return new Call(new TIR.CPU.Cast(newType, castMode), input, output);
    }

    public static Call Where(Expr cond, Expr x, Expr y, Expr output, DistributedType distributedType)
    {
        return new Call(new TIR.CPU.Where(distributedType), cond, x, y, output);
    }

    public static Call Expand(Expr input, Expr output, IRArray<int> shape, IRArray<SBP> ndsbp)
    {
        return new Call(new TIR.CPU.Expand(shape, ndsbp), input, output);
    }
}
