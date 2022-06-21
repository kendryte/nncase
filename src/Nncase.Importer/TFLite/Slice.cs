// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.IR;
using F = Nncase.IR.F;

namespace Nncase.Importer.TFLite
{
    public partial class TFLiteImporter
    {
        private Expr VisitSlice(in tflite.Operator op)
        {
            var input = GetInputExprs(op, 0);
            var (begin, size) = GetInputExprs(op, 1, 2);
            var end = begin + size;
            var count = GetInputTensor(op, 1).Shape(0);
            return F.Tensors.Slice(input, begin, end, count);
        }

        private Expr VisitStrideSlice(in tflite.Operator op)
        {
            var (input, begin) = GetInputExprs(op, 0, 1);
            var (end, strides) = GetInputExprs(op, 2, 3);
            var options = op.BuiltinOptionsAsStridedSliceOptions();
            var tensor = GetInputTensor(op, 0);
            var axes = Tensor.FromSpan<int>(Enumerable.Range(0, tensor.ShapeLength).ToArray());
            if ((options.NewAxisMask + options.ShrinkAxisMask + options.EllipsisMask) != 0)
            {
                throw new NotImplementedException("AxisMask and Ellipisis mask not impl in StrideSlice Importer");
            }
            var maskBegin = WithMask(begin, options.BeginMask, 0);
            var maskEnd = WithMask(end, options.EndMask, Int32.MaxValue);
            return F.Tensors.Slice(input, maskBegin, maskEnd, axes, strides);
        }

        private Expr WithMask(Expr data, int mask, int defaultValue)
        {
            if (mask == 0)
            {
                return data;
            }
            if (data is TensorConst constValue)
            {
                var arr = constValue.Value.ToArray<int>();
                for (int i = 0; i < arr.Length; i++)
                {
                    arr[i] = ((mask >> i) & 1) != 0
                        ? defaultValue
                        : arr[i];
                }
                return arr;
            }

            throw new NotSupportedException("StrideSlice not supported mask != 0 && dynamic data");
        }
    }
}