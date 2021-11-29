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
            var tensor = GetInputTensor(op, 1);
            return F.Tensors.Slice(input, begin, end, tensor.ShapeLength);
        }

        private Expr VisitStrideSlice(in tflite.Operator op)
        {
            var (input, begin) = GetInputExprs(op, 0, 1);
            var (end, strides) = GetInputExprs(op, 2, 3);
            var options = op.BuiltinOptionsAsStridedSliceOptions();
            if (options.BeginMask != 0 || options.EndMask != 0 || options.EllipsisMask != 0
                || options.NewAxisMask != 0)
            {
                throw new NotSupportedException("Unsupported StrideSlice no 0 mask");
            }

            var tensor = GetInputTensor(op, 0);
            var axes = Const.FromSpan<int>(Enumerable.Range(0, tensor.ShapeLength).ToArray());
            return F.Tensors.Slice(input, begin, end, axes, strides);
        }
    }
}