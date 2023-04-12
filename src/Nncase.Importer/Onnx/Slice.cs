// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using LanguageExt.UnsafeValueAccess;
using Nncase.IR;
using Nncase.IR.Tensors;
using Onnx;
using static Nncase.IR.F.Tensors;
using F = Nncase.IR.F;

namespace Nncase.Importer
{
    public partial class OnnxImporter
    {
        private Expr VisitSlice(in NodeProto op)
        {
            return GetOpSet(op) < 10
                ? SliceV1(op)
                : SliceV10(op);
        }

        private Expr SliceV1(in NodeProto op)
        {
            var input = GetSingleInputExpr(op);
            var axesExpr = GetAxesAttribute(op, input);
            var starts = GetTensorIntsAttribute(op, "starts");
            var ends = GetTensorIntsAttribute(op, "ends");
            return Slice(
                input,
                starts,
                ends,
                axesExpr,
                Expand(1L, Tensor.From<long>(new long[] { ends.Length })));
        }

        private Expr SliceV10(in NodeProto op)
        {
            var input = GetInputExpr(op, 0);
            var (starts, ends) = GetInputExprs(op, 1, 2);

            // not supported none step when starts is dynamic
            // steps.size should eq starts.size
            starts.InferenceType();
            var axes = GetOptionInputExpr(op, 3).Or(ComputeDefaultAxes(input));
            var steps = GetOptionInputExpr(op, 4).Or(Expand(1, starts.CheckedShape));
            return Slice(input, starts, ends, axes, steps);
        }

        private Call ExpandOneToRank(Expr input, long value, long rankOffset = 0)
        {
            return Expand(value, Unsqueeze(Cast(Rank(input) - rankOffset, new Int64Type()), new[] { 0 }));
        }
    }
}
