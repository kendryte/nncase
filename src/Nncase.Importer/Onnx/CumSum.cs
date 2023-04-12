// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Onnx;
using F = Nncase.IR.F;

namespace Nncase.Importer
{
    public partial class OnnxImporter
    {
        private Expr VisitCumSum(in NodeProto op)
        {
            var (input, axis) = GetInputExprs(op, 0, 1);
            var exclusive = GetBoolAttribute(op, "exclusive", false);
            var reverse = GetBoolAttribute(op, "reverse", false);
            return F.Tensors.CumSum(input, axis, exclusive, reverse);
        }
    }
}
