// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Onnx;
using static Nncase.IR.F.Tensors;

namespace Nncase.Importer
{
    public partial class OnnxImporter
    {
        private Expr VisitOneHot(in NodeProto op)
        {
            var (indices, depth) = GetInputExprs(op, 0, 1);
            var values = GetInputExpr(op, 2);
            var axis = GetIntAttribute(op, "axis", -1);
            return OneHot(OneHotMode.Normal, indices, depth, SliceIndex(values, 1), SliceIndex(values, 0), axis);
        }
    }
}