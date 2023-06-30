// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Runtime.InteropServices;
using static Nncase.IR.F.Tensors;
using Cast = Nncase.IR.Tensors.Cast;

namespace Nncase.IR
{
    public class Util
    {
        public static int PositiveIndex(int index, TensorType input)
        {
            return PositiveIndex(index, input.Shape.Rank);
        }

        public static int PositiveIndex(int index, int rank)
        {
            return index < 0 ? index + rank : index;
        }

        public static Expr GetWindowedOutputSize(Expr size, Expr filter, Expr stride, Expr dilation, bool same, bool ceilMode)
        {
            var effectiveFilterSize = ((filter - 1) * dilation) + 1;
            var falseBranch = !ceilMode
                ? ((size - effectiveFilterSize + stride) / stride)
                : F.Tensors.Cast(
                    F.Math.Ceil(
                        F.Tensors.Cast(size - effectiveFilterSize + stride, DataTypes.Float32) /
                        F.Tensors.Cast(stride, DataTypes.Float32)),
                    DataTypes.Int32);
            var trueBranch = (size + stride - 1) / stride;
            return same ? trueBranch : falseBranch;
        }

        public static Expr GetConvTransposeOutputShape(Expr inShape, Expr wShape, Expr strides, Expr outPadding, Expr paddings, Expr dilations, string autoPad, Expr group)
        {
            inShape = Cast(inShape, DataTypes.Int64);
            wShape = Cast(wShape, DataTypes.Int64);
            var iN = inShape[0];
            _ = inShape[1];
            var iH = inShape[2];
            var iW = inShape[3];
            var oc = wShape[0] * group;
            var wH = wShape[2];
            var wW = wShape[3];
            var outShape = new List<Expr>();
            outShape.Add(iN);
            outShape.Add(oc);
            if (autoPad is "SAME_UPPER" or "SAME_LOWER")
            {
                outShape.Add(iH * strides[0]);
                outShape.Add(iW * strides[1]);
            }
            else
            {
                outShape.Add(ComputeOutSize(iH, wH, strides, outPadding, paddings, dilations, 0));
                outShape.Add(ComputeOutSize(iW, wW, strides, outPadding, paddings, dilations, 1));
            }

            return F.Tensors.Stack(new IR.Tuple(CollectionsMarshal.AsSpan(outShape)), 0);
        }

        private static Expr ComputeOutSize(Expr inputSize, Expr weightSize, Expr strides, Expr outPaddings, Expr paddings, Expr dilations, int offset)
        {
            return (strides[offset] * (inputSize - 1L))
                + outPaddings[offset]
                + (((weightSize - 1L)
                    * dilations[offset]) + 1L) - paddings[offset][0] - paddings[offset][1];
        }
    }
}
