// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
using Nncase.IR;
using Nncase.IR.Tensors;
using static Nncase.IR.F.NN;
using static Nncase.IR.F.Tensors;

namespace Nncase.Importer.TFLite
{
    public partial class TFLiteImporter
    {
        private Expr VisitSpaceToBatchND(in tflite.Operator op)
        {
            var (input, blockShape) = GetInputExprs(op, 0, 1);
            var paddings = GetInputExprs(op, 2);
            return SpaceToBatch(input, blockShape, paddings);
        }

        private Expr VisitBatchToSpaceND(in tflite.Operator op)
        {
            var (input, blockShape) = GetInputExprs(op, 0, 1);
            var crops = GetInputExprs(op, 2);
            return NCHWToNHWC(BatchToSpace(NHWCToNCHW(input), blockShape, crops));
        }
    }
}
