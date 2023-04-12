// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Onnx;
using F = Nncase.IR.F;

namespace Nncase.Importer
{
    public partial class OnnxImporter
    {
        private Expr VisitReverseSequence(in NodeProto op)
        {
            var (input, seqLens) = GetInputExprs(op, 0, 1);
            var batchAxis = GetBoolAttribute(op, "batch_axis", true);
            var timeAxis = GetBoolAttribute(op, "time_axis", false);
            return F.Tensors.ReverseSequence(input, seqLens, batchAxis, timeAxis);
        }
    }
}
