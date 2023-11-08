// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Linq;
using Nncase.IR;
using Onnx;
using F = Nncase.IR.F;

namespace Nncase.Importer
{
    public partial class OnnxImporter
    {
        private Expr VisitConcat(NodeProto op)
        {
            var inputs = Enumerable.Range(0, op.Input.Count).Select(x => GetInputExpr(op, x)).ToArray();
            var axis = GetIntAttribute(op, "axis");
            return F.Tensors.Concat(new Tuple(inputs), (int)axis);
        }
    }
}
