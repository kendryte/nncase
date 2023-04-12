// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Onnx;
using static Nncase.IR.F.Tensors;

namespace Nncase.Importer
{
    public partial class OnnxImporter
    {
        private Expr VisitSqueeze(in NodeProto op)
        {
            return GetOpSet(op) < 13
                ? SqueezeV11(op)
                : SqueezeV13(op);
        }

        private Expr SqueezeV11(in NodeProto op)
        {
            var input = GetInputExpr(op, 0);
            var axes = GetOptionIntsAttribute(op, "axes").Or(System.Array.Empty<long>());
            return Squeeze(input, axes);
        }

        private Expr SqueezeV13(in NodeProto op)
        {
            var input = GetInputExpr(op, 0);
            var axes = GetOptionInputExpr(op, 1, Tensor.From<long>(System.Array.Empty<long>()));
            return Squeeze(input, axes);
        }
    }
}
