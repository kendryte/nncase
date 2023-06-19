// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.IR;
using Nncase.IR.Tensors;
using static Nncase.IR.F.Tensors;
using F = Nncase.IR.F;
using Tuple = Nncase.IR.Tuple;

namespace Nncase
{
    public static class Util
    {
        public static Expr DynamicShapeIndex(in Expr input, Expr index) => GetItem(F.Tensors.ShapeOf(input), index);

        public static Expr ShapeIndex(in Expr input, int index)
        {
            Expr i;
            if (input.CheckedType != null)
            {
                i = index < 0 ? index + input.CheckedShape.Rank : index;
            }
            else
            {
                i = index;
            }

            return DynamicShapeIndex(input, i);
        }

        public static Expr GetItem(in Expr input, Expr index)
        {
            return F.Tensors.Squeeze(
                F.Tensors.Slice(
                    input,
                    StackScalar(index),
                    StackScalar(index + 1),
                    1),
                new[] { 0L });
        }

        public static (Expr H, Expr W) GetHW(in Expr input, bool isNHWC = false)
        {
            if (isNHWC)
            {
                return (ShapeIndex(input, 1), ShapeIndex(input, 2));
            }

            return (ShapeIndex(input, 2), ShapeIndex(input, 3));
        }

        /// <summary>
        /// onnx format pads to nncase format(same as tf).
        /// </summary>
        public static Expr PadTranslate(Expr pads)
        {
            return Transpose(Reshape(pads, new[] { -1, 2 }), new[] { 1, 0 });
        }

        public static TensorConst ZeroTensor()
        {
            return new TensorConst(Tensor.From<int>(new[] { 0 }));
        }

        /// <param name="padH">H [before, after]. </param>
        /// <param name="padW">W [before, after]. </param>
        public static Expr ConcatPadding(Expr[] padH, Expr[] padW)
        {
            // return [[padh_before, padh_after],
            //         [padw_before, padw_after]]
            return Stack(
                new Tuple(
                    Stack(new Tuple(padH), 0),
                    Stack(new Tuple(padW), 0)),
                0);
        }

        // todo:refactor and set private this
        public static Expr[] GetWindowedPadding(Expr inputSize, Expr filter, Expr stride, Expr dilation, bool same, bool lower = false)
        {
            var i32InputSize = Cast(inputSize, DataTypes.Int32);
            var i32Filter = Cast(filter, DataTypes.Int32);
            var i32Stride = Cast(stride, DataTypes.Int32);
            var i32Dilation = Cast(dilation, DataTypes.Int32);
            var outputSize = IR.Util.GetWindowedOutputSize(i32InputSize, i32Filter, i32Stride, i32Dilation, same, false);
            return GetWindowedPaddingValue(i32InputSize, outputSize, i32Filter, i32Stride, i32Dilation, lower);
        }

        // lower used for onnx when auto_pad attr is SAME_LOWER
        public static Expr GetPaddings(Expr input, Expr weights, long[] stride, long[] dilation, bool same, bool lower = false)
        {
            var (inH, inW) = GetHW(input);
            var (fH, fW) = GetHW(weights);
            var padH = GetWindowedPadding(inH, fH, (int)stride[0], (int)dilation[0], same, lower);
            var padW = GetWindowedPadding(inW, fW, (int)stride[1], (int)dilation[1], same, lower);
            return ConcatPadding(padH, padW);
        }

        public static Expr ComputeSplit(Expr input, long outputSize, long axis)
        {
            return F.Tensors.Expand(
                Util.ShapeIndex(input, (int)axis) / outputSize, // Util.DynamicShapeIndex(input, Cast(axis, DataTypes.Int32)) / outputSize
                Stack(new Tuple(outputSize), 0));
        }

        private static Expr[] GetWindowedPaddingValue(Expr inputSize, Expr outputSize, Expr filter, Expr stride, Expr dilation, bool lower)
        {
            var effectiveFilterSize = ((filter - 1) * dilation) + 1;
            var padding = F.Math.Max(0, ((outputSize - 1) * stride) + effectiveFilterSize - inputSize);
            var before = F.Tensors.Cast(padding / 2, DataTypes.Int32);
            var after = F.Tensors.Cast(padding - (padding / 2), DataTypes.Int32);
            if (lower)
            {
                return new[] { F.Math.Max(before, after), F.Math.Min(before, after) };
            }

            return new[] { before, after };
        }
    }
}
