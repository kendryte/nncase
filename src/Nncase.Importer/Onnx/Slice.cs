// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using LanguageExt.UnsafeValueAccess;
using Nncase.IR;
using Nncase.IR.Tensors;
using Onnx;
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
            Expr axesExpr = GetAxesAttribute(op, input);
            var starts = GetConstIntsAttribute(op, "starts");
            var ends = GetConstIntsAttribute(op, "ends");
            return F.Tensors.Slice(input, starts, ends, axesExpr, ExpandOneToRank(input, 1));
        }
        
        private Expr SliceV10(in NodeProto op)
        {
            var input = GetInputExpr(op, 0);
            var (starts, ends) = GetInputExprs(op, 1, 2);
            var axes = GetOptionInputExpr(op, 3).Or(ComputeDefaultAxes(input));
            var steps = GetOptionInputExpr(op, 4).Or(ExpandOneToRank(input, 1));
            return F.Tensors.Slice(input, starts, ends, axes, steps);
        }

        private Call ExpandOneToRank(Expr input, Expr value, int rankOffset = 0)
        {
            return F.Tensors.Expand(value, F.Tensors.Rank(input) - rankOffset);
        }
    }
}