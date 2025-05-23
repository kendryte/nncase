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
            var input = GetSingleInputExpr<Expr>(op);
            var axesExpr = GetAxesAttribute(op, input);
            var starts = GetIntsAttribute(op, "starts");
            var ends = GetIntsAttribute(op, "ends");
            return Slice(
                input,
                starts,
                ends,
                axesExpr,
                Shape.Repeat(1L, ends.Length));
        }

        private Expr SliceV10(in NodeProto op)
        {
            var input = GetInputExpr<Expr>(op, 0);
            var (starts, ends) = GetInputExprs<Shape, Shape>(op, 1, 2);

            // not supported none step when starts is dynamic
            // steps.size should eq starts.size
            starts.InferenceType();
            var axes = GetOptionInputExpr<Shape>(op, 3).Or(ComputeDefaultAxes(input));
            var steps = GetOptionInputExpr<Shape>(op, 4).Or(Shape.Repeat(1, starts.Rank));
            return Slice(input, starts, ends, axes, steps);
        }
    }
}
