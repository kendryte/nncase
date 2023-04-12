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
        private Expr VisitGatherND(in NodeProto op)
        {
            var (input, indices) = GetInputExprs(op, 0, 1);
            var batchDims = GetIntAttribute(op, "batch_dims", 0);
            return F.Tensors.GatherND(input, batchDims, indices);
        }
    }
}
