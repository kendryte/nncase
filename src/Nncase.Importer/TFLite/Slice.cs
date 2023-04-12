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
            var (input, beginExpr) = GetInputExprs(op, 0, 1);
            var (endExpr, strides) = GetInputExprs(op, 2, 3);
            var options = op.BuiltinOptionsAsStridedSliceOptions();
            var tensor = GetInputTensor(op, 0);
            var axes = Tensor.From<int>(Enumerable.Range(0, tensor.ShapeLength).ToArray());
            if ((options.NewAxisMask + options.EllipsisMask) != 0)
            {
                throw new NotImplementedException("NewAxisMask and Ellipisis mask not impl in StrideSlice Importer");
            }

            var beginMask = options.BeginMask;
            var endMask = options.EndMask;
            var shrinkMask = options.ShrinkAxisMask;
            if ((beginMask + endMask + shrinkMask) == 0)
            {
                return F.Tensors.Slice(input, beginExpr, endExpr, axes, strides);
            }

            if (beginExpr is TensorConst beginConst && endExpr is TensorConst endConst)
            {
                var begin = beginConst.Value.ToArray<int>();
                var end = endConst.Value.ToArray<int>();
                var newBegin = new List<int>();
                var newEnd = new List<int>();
                var ellipsisGap = 0;
                var needSqueeze = new List<int>();
                for (int i = 0; i < begin.Length; i++)
                {
                    var beginItem = begin[i];
                    var endItem = end[i];
                    if (((shrinkMask >> i) & 1) != 0)
                    {
                        newBegin.Add(beginItem);
                        newEnd.Add(beginItem != -1 ? beginItem + 1 : int.MaxValue);
                        needSqueeze.Add(i + ellipsisGap);
                        continue;
                    }

                    newBegin.Add(((beginMask >> i) & 1) != 0 ? 0 : beginItem);
                    newEnd.Add(((endMask >> i) & 1) != 0 ? int.MaxValue : endItem);
                }

                var result = F.Tensors.Slice(input, newBegin.ToArray(), newEnd.ToArray(), axes, strides);
                if (needSqueeze.Count != 0)
                {
                    result = F.Tensors.Squeeze(result, needSqueeze.ToArray());
                }

                return result;
            }
            else
            {
                throw new NotSupportedException("StrideSlice not supported mask != 0 && dynamic data");
            }
        }
    }
}
