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

    public static Call Unary(string unaryOp, Expr input, Expr output)
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

    public static Call InstanceNorm(float eps, Expr input, Expr scale, Expr bias, Expr output, DistributedType distributedType)
    {
        return new Call(new InstanceNorm(eps, distributedType), input, scale, bias, output);
    }

    public static Call Gather(int axis, Expr input, Expr indices, Expr output)
    {
        return new Call(new IR.XPU.Gather(axis), input, indices, output);
    }

    public static Call Concat(int axis, Expr[] inputs, Expr output)
    {
        return new Call(new Concat(axis), inputs.Concat(new[] { output }).ToArray());
    }

    public static Expr Slice(TIR.Buffer input, TIR.Buffer output, Expr begins, Expr ends, Expr axes, DistributedType distributedType)
    {
        return new Call(new Slice(distributedType), input, output, begins, ends, axes);
    }

    public static Call Softmax(int axis, Expr input, Expr output, DistributedType distributedType)
    {
        return new Call(new IR.XPU.Softmax(axis, distributedType), input, output);
    }

    public static Call Transpose(int[] perm, Expr input, Expr output)
    {
        return new Call(new IR.XPU.Transpose(perm), input, output);
    }

    public static Call ReShape(Expr input, Expr output)
    {
        return new Call(new IR.XPU.ReShape(), input, output);
    }

    public static Call GatherReduceScatter(Expr input, Expr output, (int, SBP)[] reducePosition, Placement placement)
    {
        return new Call(new GatherReduceScatter(reducePosition, placement), input, output);
    }

    public static Call Conv2D(Expr input, Expr weights, Expr bias, Expr output, int[] stride, int[] padding, int[] dilation, int groups, TensorConst fusedClamp, DistributedType distributedType)
    {
        return new Call(new Conv2D(stride, padding, dilation, groups, fusedClamp, distributedType), input, weights, bias, output);
    }

    public static Call ReduceArg(Expr input, Expr output, int axis, bool keepdims, bool selectLastIndex, ReduceArgOp op, DataType dataType)
    {
        return new Call(new ReduceArg(axis, keepdims, selectLastIndex, op, dataType), input, output);
    }

    public static Call Resize(Expr input, Expr output, float[] roi, int[] newSize, float cubicCoeffA, int excludeOutside, float extrapolationValue, ImageResizeMode resizeMode, ImageResizeTransformationMode transformationMode, ImageResizeNearestMode nearestMode, bool isTFResize)
    {
        return new Call(new Resize(roi, newSize, cubicCoeffA, excludeOutside, extrapolationValue, resizeMode, transformationMode, nearestMode, isTFResize), input, output);
    }

    public static Call Cast(Expr input, Expr output, DataType dataType, CastMode castMode)
    {
        return new Call(new Cast(dataType, castMode), input, output);
    }

    public static Call Expand(int[] shape, DistributedType distributedType, Expr input, Expr output)
    {
        return new Call(new Expand(shape, distributedType.NdSBP), input, output);
    }
}
