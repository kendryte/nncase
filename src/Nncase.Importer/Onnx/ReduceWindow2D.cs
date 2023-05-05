// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Linq;
using System.Security.Cryptography;
using Nncase.IR;
using Nncase.IR.Tensors;
using Onnx;
using F = Nncase.IR.F;

namespace Nncase.Importer
{
    public partial class OnnxImporter
    {
        // isGlobal used for GlobalXXXPool
        private Expr VisitReduceWindow2D(in NodeProto op, ReduceOp reduceOp, float initValue, bool isGlobal = false)
        {
            // auto_pad had been DEPRECATED
            var input = GetInputExpr(op, 0);
            if (isGlobal && input.CheckedShape.Rank != 4)
            {
                return F.Tensors.Reduce(reduceOp, input, Enumerable.Range(0, input.CheckedShape.Rank).Skip(2).ToArray(), initValue, true);
            }

            var ceilMode = GetBoolAttribute(op, "ceil_mode", false);
            var countIncludePad = GetBoolAttribute(op, "count_include_pad", false);
            var pads = GetPadsAttribute(op);
            var dilation = reduceOp == ReduceOp.Max
                ? GetIntsAttribute(op, "dilations", 1, 2)
                : Enumerable.Repeat<long>(1, 2).ToArray();
            var kernelShape = isGlobal
                ? Util.GetHW(input).Map((h, w) => (Expr)F.Tensors.Stack(new Tuple(h, w), 0))
                : Tensor.From<long>(GetIntsAttribute(op, "kernel_shape"));
            var strides = GetStrideAttribute(op);
            return F.NN.ReduceWindow2D(
                reduceOp,
                input,
                initValue,
                kernelShape,
                strides,
                pads,
                Tensor.From<long>(dilation),
                ceilMode,
                countIncludePad);
        }
    }
}
