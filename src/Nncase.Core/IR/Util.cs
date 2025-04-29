// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Runtime.InteropServices;
using Nncase.IR.Shapes;
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

        public static RankedShape GetConvTransposeOutputShape(Shape inShape, Shape wShape, Shape strides, Shape outPadding, Paddings paddings, Shape dilations, string autoPad, Dimension group)
        {
            var iN = inShape[0];
            _ = inShape[1];
            var iH = inShape[2];
            var iW = inShape[3];
            var oc = wShape[0] * group;
            var wH = wShape[2];
            var wW = wShape[3];
            var outShape = new List<Dimension>();
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

            return new RankedShape(CollectionsMarshal.AsSpan(outShape));
        }

        private static Dimension ComputeOutSize(Dimension inputSize, Dimension weightSize, Shape strides, Shape outPaddings, Paddings paddings, Shape dilations, int offset)
        {
            return (strides[offset] * (inputSize - 1L))
                + outPaddings[offset]
                + (((weightSize - 1L)
                    * dilations[offset]) + 1L) - paddings[offset].Sum();
        }
    }
}
