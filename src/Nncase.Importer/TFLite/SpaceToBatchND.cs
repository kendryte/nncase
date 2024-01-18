// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
using Nncase.IR;
using Nncase.IR.Tensors;
using static Nncase.IR.F.NN;
using static Nncase.IR.F.Tensors;
using Unsqueeze = Nncase.IR.Tensors.Unsqueeze;

namespace Nncase.Importer.TFLite
{
    public partial class TFLiteImporter
    {
        private Expr VisitSpaceToBatchND(in tflite.Operator op)
        {
            var (input, blockShape) = GetInputExprs(op, 0, 1);
            var paddings = GetInputExprs(op, 2);
            if (input.CheckedShape.Rank == 3)
            {
                blockShape = Concat(new IR.Tuple(new[] { new[] { 1 }, blockShape }), 0);
                paddings = Concat(new IR.Tuple(new[] { new[,] { { 0, 0 } }, paddings }), 0);
                input = Unsqueeze(input, new[] { -3 });
            }

            var stb = SpaceToBatch(NHWCToNCHW(input), blockShape, paddings);
            if (input.CheckedShape.Rank == 4)
            {
                stb = Squeeze(stb, new[] { 1 });
            }

            return NCHWToNHWC(stb);
        }

        private Expr VisitBatchToSpaceND(in tflite.Operator op)
        {
            var (input, blockShape) = GetInputExprs(op, 0, 1);
            var crops = GetInputExprs(op, 2);
            if (input.CheckedShape.Rank == 3)
            {
                blockShape = Concat(new IR.Tuple(new[] { new[] { 1 }, blockShape }), 0);
                crops = Concat(new IR.Tuple(new[] { new[,] { { 0, 0 } }, crops }), 0);
                input = Unsqueeze(input, new[] { -3 });
            }

            var bts = BatchToSpace(NHWCToNCHW(input), blockShape, crops);
            if (input.CheckedShape.Rank == 4)
            {
                bts = Squeeze(bts, new[] { 1 });
            }

            return NCHWToNHWC(bts);
        }
    }
}
