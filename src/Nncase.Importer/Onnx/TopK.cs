// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.Tensors;
using Onnx;
using static Nncase.IR.F.Tensors;

namespace Nncase.Importer
{
    public partial class OnnxImporter
    {
        private Expr VisitTopK(in NodeProto op)
        {
            var (x, k) = GetInputExprs(op, 0, 1);
            var axis = GetOptionIntAttribute(op, "axis").Or(-1);
            var largest = GetOptionIntAttribute(op, "largest").Or(1);
            var sorted = GetOptionIntAttribute(op, "sorted").Or(1);
            return TopK(x, k, axis, largest, sorted);
        }
    }
}
