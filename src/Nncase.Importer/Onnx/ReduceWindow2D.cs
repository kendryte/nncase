// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Linq;
using System.Security.Cryptography;
using Nncase.IR;
using Nncase.IR.Tensors;
using Onnx;
using static Nncase.IR.F.Tensors;
using F = Nncase.IR.F;

namespace Nncase.Importer
{
    public partial class OnnxImporter
    {
        // isGlobal used for GlobalXXXPool
        private Expr VisitReduceWindow2D(in NodeProto op, ReduceOp reduceOp, Expr initValue, bool isGlobal = false)
        {
            // auto_pad had been DEPRECATED
            var input = GetInputExpr<Expr>(op, 0);
            if (isGlobal && input.CheckedShape.Rank != 4)
            {
                return F.Tensors.Reduce(reduceOp, input, Enumerable.Range(0, input.CheckedShape.Rank).Skip(2).ToArray(), initValue, true);
            }

            var ceilMode = GetBoolAttribute(op, "ceil_mode", false);
            var countIncludePad = GetBoolAttribute(op, "count_include_pad", false);
            var dilation = reduceOp == ReduceOp.Max
                ? GetIntsAttribute(op, "dilations", 1, 2).ToArray()
                : Enumerable.Repeat<long>(1, 2).ToArray();
            var kernelShape = isGlobal
                ? Util.GetHW(input).Map((h, w) => new RankedShape(h, w))
                : GetIntsAttribute(op, "kernel_shape");
            var strides = GetStrideAttribute(op).ToArray<long>().ToList();

            var isPool1D = input.CheckedShape.Rank == 3;
            var pads = GetPadsAttribute(op, isPool1D);
            if (isPool1D)
            {
                strides.Add(1);
                kernelShape = kernelShape.Append(1).ToArray();
                input = To4D(input);
            }

            var pdp = F.NN.ReduceWindow2D(
                reduceOp,
                input,
                initValue,
                kernelShape,
                strides.ToArray(),
                pads,
                dilation,
                ceilMode,
                countIncludePad);

            if (isPool1D)
            {
                pdp = Squeeze(pdp, new[] { 3 });
            }

            return pdp;
        }
    }
}
