// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.IR;
using Nncase.IR.Shapes;
using Nncase.IR.Tensors;
using tflite;
using static Nncase.IR.F.NN;
using static Nncase.IR.F.Tensors;
using Unsqueeze = Nncase.IR.Tensors.Unsqueeze;

namespace Nncase.Importer.TFLite
{
    public partial class TFLiteImporter
    {
        private Expr VisitSpaceToBatchND(in tflite.Operator op)
        {
            var (input, blockShape) = GetInputExprs<Expr, RankedShape>(op, 0, 1);
            var paddings = GetInputExprs<Paddings>(op, 2).AsPaddings();
            bool needUnsqueeze = input.CheckedShape.Rank == 3;
            if (needUnsqueeze)
            {
                blockShape = new RankedShape([1, .. blockShape]);
                paddings = new Paddings([IR.Shapes.Padding.Zero, .. paddings]);
                input = Unsqueeze(input, new[] { -3 });
            }

            var stb = NCHWToNHWC(SpaceToBatch(NHWCToNCHW(input), blockShape, paddings));
            if (needUnsqueeze)
            {
                return Squeeze(stb, new[] { 1 });
            }

            return stb;
        }

        private Expr VisitBatchToSpaceND(in tflite.Operator op)
        {
            var (input, blockShape) = GetInputExprs<Expr, RankedShape>(op, 0, 1);
            var crops = GetInputExprs<Paddings>(op, 2);
            bool needUnsqueeze = input.CheckedShape.Rank == 3;
            if (needUnsqueeze)
            {
                blockShape = new RankedShape([1, .. blockShape]);
                crops = new Paddings([IR.Shapes.Padding.Zero, .. crops]);
                input = Unsqueeze(input, new[] { -3 });
            }

            var bts = NCHWToNHWC(BatchToSpace(NHWCToNCHW(input), blockShape, crops));
            if (needUnsqueeze)
            {
                return Squeeze(bts, new[] { 1 });
            }

            return bts;
        }
    }
}
