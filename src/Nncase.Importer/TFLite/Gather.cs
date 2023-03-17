// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using F = Nncase.IR.F;

namespace Nncase.Importer.TFLite
{
    public partial class TFLiteImporter
    {
        private Expr VisitGather(in tflite.Operator op)
        {
            var (input, indices) = GetInputExprs(op, 0, 1);
            return SetOutputsNames(
                F.Tensors.Gather(input, op.BuiltinOptionsAsGatherOptions().Axis, indices),
                1,
                op);
        }

        private Expr VisitGatherND(in tflite.Operator op)
        {
            var (input, indices) = GetInputExprs(op, 0, 1);
            return SetOutputsNames(F.Tensors.GatherND(input, 0, indices), 1, op);
        }
    }
}
