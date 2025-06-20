﻿// Copyright (c) Canaan Inc. All rights reserved.
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

    public static Expr PackedSoftmax(Expr input, int axis, IRArray<int> packedAxes)
    {
        return new Call(new PackedSoftmax(axis, packedAxes), input);
    }

    public static Expr PackedLayerNorm(Expr input, Expr scale, Expr bias, int axis, float epsilon, bool usemean, IRArray<int> packedAxes, IRArray<int> padedNums)
    {
        return new Call(new PackedLayerNorm(axis, epsilon, usemean, packedAxes, padedNums), input, scale, bias);
    }

    public static Call PackedReduce(Expr input, ReduceOp reduceOp, IRArray<int> axes, float initValue, bool keepDims, IRArray<int> packedAxes, IRArray<int> padedNums)
    {
        return new Call(new PackedReduce(reduceOp, axes, initValue, keepDims, packedAxes, padedNums), input);
    }

    public static Expr InstacneNorm(Expr input, Expr scale, Expr bias, float epsilon, IRArray<int> packedAxes, IRArray<int> padedNums)
    {
        return new Call(new InstacneNorm(epsilon, packedAxes, padedNums), input, scale, bias);
    }

    public static Expr PackedMatMul(Expr lhs, Expr rhs, IRArray<int> lhsPackedAxes, IRArray<int> rhsPackedAxes, bool transA = false, bool transB = false, bool fusedReduce = false)
    {
        return new Call(new PackedMatMul(DataTypes.Float32, lhsPackedAxes, rhsPackedAxes, transA, transB, fusedReduce), lhs, rhs);
    }

    public static Expr PackedMatMul(Expr lhs, Expr rhs, IRArray<int> lhsPackedAxes, IRArray<int> rhsPackedAxes, bool transA = false, bool transB = false, bool fusedReduce = false, DataType outDataType = null!)
    {
        return new Call(new PackedMatMul(outDataType, lhsPackedAxes, rhsPackedAxes, transA, transB, fusedReduce), lhs, rhs);
    }

    public static Expr PackedBinary(Expr lhs, Expr rhs, BinaryOp binaryOp, IRArray<int> lhsPackedAxes, IRArray<int> lhsPadedNums, IRArray<int> rhsPackedAxes, IRArray<int> rhsPadedNums)
    {
        return new Call(new PackedBinary(binaryOp, lhsPackedAxes, lhsPadedNums, rhsPackedAxes, rhsPadedNums), lhs, rhs);
    }

    public static Call ResizeImage(Expr input, int[] packedAxes, int[] padedNums, int[] newSize, ImageResizeMode resizeMode, ImageResizeTransformationMode transformationMode, ImageResizeNearestMode nearestMode)
    {
        return new Call(new ResizeImage(packedAxes, padedNums, newSize, resizeMode, transformationMode, nearestMode), input);
    }

    public static Expr Im2col(Expr input, long[] kernel, int[] stride, int[] padding)
    {
        return new Call(new Im2col(kernel, stride, padding, Array.Empty<int>(), Array.Empty<int>()), input);
    }

    public static Expr Im2col(Expr input, long[] kernel, int[] stride, int[] padding, int[] packedAxes, int[] padedNums)
    {
        return new Call(new Im2col(kernel, stride, padding, packedAxes, padedNums), input);
    }
}
