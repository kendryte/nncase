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
        private Expr VisitTrilu(in NodeProto op)
        {
            var input = GetInputExpr(op, 0);
            var k = GetOptionInputExpr(op, 1).Or(0L);
            var upper = GetIntAttribute(op, "upper", 1);
            return Trilu(input, k, upper);
        }
    }
}
