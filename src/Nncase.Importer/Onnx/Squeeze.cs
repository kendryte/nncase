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
            var input = GetInputExpr(op, 0);
            var axes = GetOptionInputExpr(op, 1, Tensor.FromSpan<long>(new long[]{}));
            return Squeeze(input, axes);
        }
    }
}