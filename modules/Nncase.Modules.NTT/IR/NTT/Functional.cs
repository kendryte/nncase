// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR.NTT;
using Nncase.IR.Tensors;

namespace Nncase.IR.F;

public partial class NTT
{
    public static Call Load(Expr input)
    {
        return new Call(new Load(), input);
    }

    public static Call Store(Expr input)
    {
        return new Call(new Store(), input);
    }

    public static Expr VectorizedSoftmax(Expr input, int axis, IRArray<int> vectorizedAxes)
    {
        return new Call(new VectorizedSoftmax(axis, vectorizedAxes), input);
    }

    public static Expr VectorizedLayerNorm(Expr input, Expr scale, Expr bias, int axis, float epsilon, bool usemean, IRArray<int> vectorizedAxes, BaseExpr padedNums)
    {
        return new Call(new VectorizedLayerNorm(axis, epsilon, usemean, vectorizedAxes), input, scale, bias, padedNums);
    }

    public static Call VectorizedReduce(Expr input, ReduceOp reduceOp, IRArray<int> axes, float initValue, bool keepDims, IRArray<int> vectorizedAxes, BaseExpr padedNums)
    {
        return new Call(new VectorizedReduce(reduceOp, axes, initValue, keepDims, vectorizedAxes), input, padedNums);
    }

    public static Expr InstacneNorm(Expr input, Expr scale, Expr bias, float epsilon, IRArray<int> vectorizedAxes, BaseExpr padedNums)
    {
        return new Call(new InstacneNorm(epsilon, vectorizedAxes), input, scale, bias, padedNums);
    }

    public static Expr PackedMatMul(Expr lhs, Expr rhs, bool fusedReduce = false, DataType? outDataType = null)
    {
        return new Call(new PackedMatMul(outDataType ?? DataTypes.Float32, fusedReduce), lhs, rhs);
    }

    public static Expr VectorizedMatMul(Expr lhs, Expr rhs, IRArray<int> lhsVectorizedAxes, IRArray<int> rhsVectorizedAxes, bool transA = false, bool transB = false, bool fusedReduce = false, DataType? outDataType = null)
    {
        return new Call(new VectorizedMatMul(outDataType ?? DataTypes.Float32, lhsVectorizedAxes, rhsVectorizedAxes, transA, transB, fusedReduce), lhs, rhs);
    }

    public static Expr VectorizedBinary(Expr lhs, Expr rhs, BinaryOp binaryOp, IRArray<int> lhsVectorizedAxes, IRArray<Dimension> lhsPadedNums, IRArray<int> rhsVectorizedAxes, IRArray<Dimension> rhsPadedNums)
    {
        return new Call(new VectorizedBinary(binaryOp, lhsVectorizedAxes, lhsPadedNums, rhsVectorizedAxes, rhsPadedNums), lhs, rhs);
    }

    public static Call ResizeImage(Expr input, BaseExpr paddedNums, int[] vectorizedAxes, int[] newSize, ImageResizeMode resizeMode, ImageResizeTransformationMode transformationMode, ImageResizeNearestMode nearestMode)
    {
        return new Call(new ResizeImage(vectorizedAxes, newSize, resizeMode, transformationMode, nearestMode), input, paddedNums);
    }

    public static Expr Im2col(Expr input, long[] kernel, int[] stride, int[] padding)
    {
        return new Call(new Im2col(kernel, stride, padding, Array.Empty<int>(), Array.Empty<int>()), input);
    }

    public static Expr Im2col(Expr input, long[] kernel, int[] stride, int[] padding, int[] vectorizedAxes, int[] padedNums)
    {
        return new Call(new Im2col(kernel, stride, padding, vectorizedAxes, padedNums), input);
    }
}
