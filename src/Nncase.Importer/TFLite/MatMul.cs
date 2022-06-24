// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.IO;
using Nncase.IR;
using Nncase.IR.Tensors;
using tflite;
using F = Nncase.IR.F;
using static Nncase.IR.F.Tensors;
using TensorType = tflite.TensorType;

namespace Nncase.Importer.TFLite
{
    public partial class TFLiteImporter
    {
        private Expr VisitMatMul(in tflite.Operator op)
        {
            var (input, other) = GetInputExprs(op, 0, 1);
            var inTensor = GetInputTensor(op, 0);
            var otherTensor = GetInputTensor(op, 1);
            var options = op.BuiltinOptionsAsFullyConnectedOptions();
            if (options.WeightsFormat !=
                tflite.FullyConnectedOptionsWeightsFormat.DEFAULT)
            {
                throw new NotSupportedException();
            }

            if (options.FusedActivationFunction != ActivationFunctionType.NONE)
            {
                throw new NotImplementedException();
            }

            if (inTensor.Type != TensorType.FLOAT32 || otherTensor.Type != TensorType.FLOAT32)
            {
                throw new NotImplementedException();
            }

            var lhs = input;
            if (inTensor.ShapeLength != 2)
            {
                if (otherTensor.ShapeLength != 2)
                {
                    throw new NotSupportedException();
                }

                lhs = Reshape(lhs, new[] { -1, otherTensor.Shape(1) });
            }

            if (otherTensor.ShapeLength > 2)
            {
                throw new NotSupportedException("rhs rank > 2");
            }
            // todo:fused clamp
            var rhs = Transpose(other, new[] { 1, 0 });
            var bias = op.InputsLength == 3 && op.Inputs(2) != -1
                ? GetInputExprs(op, 2)
                : Expand(Cast(0, GetDataType(GetInputTensor(op, 0).Type)), new[]{otherTensor.Shape(0)}).Evaluate().AsTensor();
            return MatMul(
                lhs,
                 rhs) + bias;
        }
    }
}