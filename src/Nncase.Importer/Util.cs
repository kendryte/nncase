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
        public static Dimension ShapeIndex(in Expr input, int index)
        {
            int i;
            if (input.CheckedType is TensorType tensorType && tensorType.Shape.IsRanked)
            {
                i = index < 0 ? index + tensorType.Shape.Rank : index;
                return tensorType.Shape[i];
            }
            else
            {
                throw new InvalidOperationException($"Expr {input} has no shape");
            }
        }

        public static (Dimension H, Dimension W) GetHW(in Expr input, bool isNHWC = false)
        {
            if (isNHWC)
            {
                return (ShapeIndex(input, 1), ShapeIndex(input, 2));
            }

            return (ShapeIndex(input, 2), ShapeIndex(input, 3));
        }

        public static TensorConst ZeroTensor()
        {
            return new TensorConst(Tensor.From<int>(new[] { 0 }));
        }

        public static Shape ComputeSplit(Expr input, long outputSize, long axis)
        {
            return Shape.Repeat(Util.ShapeIndex(input, (int)axis) / outputSize, (int)outputSize);
        }
    }
}
