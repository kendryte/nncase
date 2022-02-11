// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using F = Nncase.IR.F;

namespace Nncase.Importer.TFLite
{
    public partial class TFLiteImporter
    {
        private Expr VisitReduce(in tflite.Operator op, ReduceOp reduceOp, float initValue)
        {
            var (input, axis) = GetInputExprs(op, 0, 1);
            return F.Tensors.Reduce(reduceOp, input, axis, initValue, op.BuiltinOptionsAsReducerOptions().KeepDims);
        }
    }
}