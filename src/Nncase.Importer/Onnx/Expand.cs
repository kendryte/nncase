// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.Tensors;
using Onnx;
using F = Nncase.IR.F;

namespace Nncase.Importer
{
    public partial class OnnxImporter
    {
        private Expr VisitExpand(in NodeProto op)
        {
            // broadcast rule of expand is similar to broadcast but not same
            var (input, shape) = GetInputExprs(op, 0, 1);
            return input * F.Tensors.Broadcast(1, shape);
        }
    }
}