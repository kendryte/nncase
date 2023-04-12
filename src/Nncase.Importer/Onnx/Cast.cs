// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Onnx;
using F = Nncase.IR.F;

namespace Nncase.Importer
{
    public partial class OnnxImporter
    {
        private Expr VisitCast(in NodeProto op)
        {
            var input = GetInputExpr(op, 0);

            // op set v1 to is string
            var typeIndex = GetIntAttribute(op, "to");
            return F.Tensors.Cast(input, GetDataType((int)typeIndex));
        }
    }
}
