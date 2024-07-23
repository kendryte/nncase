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

    public static Expr Pack(Expr input, int[] lanes, int[] axes)
    {
        if (lanes.Length != axes.Length)
        {
            throw new NotSupportedException();
        }

        if (axes.Length == 0)
        {
            return input;
        }

        return new Call(new Pack(lanes, axes), input);
    }

    public static Expr Unpack(Expr input, int[] lanes, int[] axes)
    {
        if (lanes.Length != axes.Length)
        {
            throw new NotSupportedException();
        }

        if (axes.Length == 0)
        {
            return input;
        }

        return new Call(new Unpack(lanes, axes), input);
    }

    public static Expr PackedSoftmax(Expr input, int axis, IRArray<int> packedAxes)
    {
        return new Call(new PackedSoftmax(axis, packedAxes), input);
    }

    public static Expr PackedLayerNorm(Expr input, Expr scale, Expr bias, int axis, float epsilon, bool usemean, IRArray<int> packedAxes, IRArray<int> padedNums)
    {
        return new Call(new PackedLayerNorm(axis, epsilon, usemean, packedAxes, padedNums), input, scale, bias);
    }

    public static Expr InstacneNorm(Expr input, Expr scale, Expr bias, float epsilon, IRArray<int> packedAxes, IRArray<int> padedNums)
    {
        return new Call(new InstacneNorm(epsilon, packedAxes, padedNums), input, scale, bias);
    }

    public static Expr PackedMatMul(Expr lhs, Expr rhs, IRArray<int> lhsPackedAxes, IRArray<int> lhsPadedNums, IRArray<int> rhsPackedAxes, IRArray<int> rhsPadedNums)
    {
        return new Call(new PackedMatMul(lhsPackedAxes, lhsPadedNums, rhsPackedAxes, rhsPadedNums), lhs, rhs);
    }

    public static Expr PackedBinary(Expr lhs, Expr rhs, BinaryOp binaryOp, IRArray<int> lhsPackedAxes, IRArray<int> lhsPadedNums, IRArray<int> rhsPackedAxes, IRArray<int> rhsPadedNums)
    {
        return new Call(new PackedBinary(binaryOp, lhsPackedAxes, lhsPadedNums, rhsPackedAxes, rhsPadedNums), lhs, rhs);
    }

    public static Call ResizeImage(Expr input, int[] packedAxes, int[] padedNums, int[] newSize, ImageResizeMode resizeMode, ImageResizeTransformationMode transformationMode, ImageResizeNearestMode nearestMode)
    {
        return new Call(new ResizeImage(packedAxes, padedNums, newSize, resizeMode, transformationMode, nearestMode), input);
    }

    public static Expr PackedTranspose(Expr input, Expr perm, IRArray<int> packedAxes)
    {
        return new Call(new PackedTranspose(packedAxes), input, perm);
    }

    public static Expr Im2col(Expr input, int[] kernel, int[] stride, int[] padding)
    {
        return new Call(new Im2col(kernel, stride, padding, Array.Empty<int>(), Array.Empty<int>()), input);
    }

    public static Expr Im2col(Expr input, int[] kernel, int[] stride, int[] padding, int[] packedAxes, int[] padedNums)
    {
        return new Call(new Im2col(kernel, stride, padding, packedAxes, padedNums), input);
    }
}
