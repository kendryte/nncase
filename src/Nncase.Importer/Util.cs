﻿// Copyright (c) Canaan Inc. All rights reserved.
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

        public static Expr ComputeSplit(Expr input, long outputSize, long axis)
        {
            return F.Tensors.Expand(
                Util.ShapeIndex(input, (int)axis) / outputSize, // Util.DynamicShapeIndex(input, Cast(axis, DataTypes.Int32)) / outputSize
                Stack(new Tuple(outputSize), 0));
        }
    }
}
