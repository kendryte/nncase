// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.IR;
using static Nncase.IR.F.Tensors;

namespace Nncase.Importer.TFLite
{
    public partial class TFLiteImporter
    {
        private Expr VisitReduce(in tflite.Operator op, ReduceOp reduceOp, float initValue)
        {
            var (input, axis) = GetInputExprs(op, 0, 1);
            return Reduce(reduceOp, input, ProcAxis(axis), initValue, op.BuiltinOptionsAsReducerOptions().KeepDims);
        }

        private Expr VisitReduceArg(in tflite.Operator op, ReduceArgOp reduceArgOp)
        {
            var (input, axis) = GetInputExprs(op, 0, 1);
            var outType = reduceArgOp switch
            {
                ReduceArgOp.ArgMin => op.BuiltinOptionsAsArgMaxOptions().OutputType,
                ReduceArgOp.ArgMax => op.BuiltinOptionsAsArgMinOptions().OutputType,
                _ => throw new ArgumentOutOfRangeException(nameof(reduceArgOp), reduceArgOp, null),
            };

            return ReduceArg(reduceArgOp, (PrimType)GetDataType(outType), input, ProcAxis(axis), false, false);
        }

        private Expr ProcAxis(Expr axis)
        {
            if (axis is TensorConst axisValue)
            {
                // scalar to array
                return axisValue.Value.ToArray<int>();
            }

            return axis;
        }
    }
}
