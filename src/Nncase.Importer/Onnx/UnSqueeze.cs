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
        private Expr VisitUnsqueeze(in NodeProto op)
        {
            return GetOpSet(op) < 13
                ? UnsqueezeV1(op)
                : UnsqueezeV13(op);
        }

        private Expr UnsqueezeV1(in NodeProto op)
        {
            var input = GetInputExpr(op, 0);
            var axes = Tensor.From<long>(GetIntsAttribute(op, "axes"));
            return Unsqueeze(input, axes);
        }

        private Expr UnsqueezeV13(in NodeProto op)
        {
            var input = GetInputExpr(op, 0);
            var axes = GetOptionInputExpr(op, 1, ComputeDefaultAxes(input));
            return Unsqueeze(input, axes);
        }
    }
}
