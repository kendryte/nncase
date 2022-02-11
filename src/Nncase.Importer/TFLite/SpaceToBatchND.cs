// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.IO;
using Nncase.IR;
using F = Nncase.IR.F;

namespace Nncase.Importer.TFLite
{
    public partial class TFLiteImporter
    {
        private Expr VisitSpaceToBatchND(in tflite.Operator op)
        {
            var (input, blockShape) = GetInputExprs(op, 0, 1);
            var paddings = GetInputExprs(op, 2);
            return F.NN.SpaceToBatch(input, blockShape, paddings);
        }

        private Expr VisitBatchToSpaceND(in tflite.Operator op)
        {
            var (input, blockShape) = GetInputExprs(op, 0, 1);
            var crops = GetInputExprs(op, 2);
            return F.NN.BatchToSpace(input, blockShape, crops);
        }
    }
}