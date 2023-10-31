// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Nncase.IR;
using Nncase.IR.NN;
using Nncase.IR.Tensors;
using tflite;
using static Nncase.IR.F.Tensors;
using F = Nncase.IR.F;
using TensorType = tflite.TensorType;

namespace Nncase.Importer.TFLite
{
    public partial class TFLiteImporter
    {
        private Expr VisitMatMul(in tflite.Operator op, bool isFullyConnected = true)
        {
            var (input, other) = GetInputExprs(op, 0, 1);
            var inTensor = GetInputTensor(op, 0);
            var otherTensor = GetInputTensor(op, 1);

            if (inTensor.Type != TensorType.FLOAT32 || otherTensor.Type != TensorType.FLOAT32)
            {
                throw new NotImplementedException();
            }

            var lhs = input;
            var rhs = other;
            var fusedActivationFunction = ActivationFunctionType.NONE;
            if (isFullyConnected)
            {
                var options = op.BuiltinOptionsAsFullyConnectedOptions();
                if (options.WeightsFormat !=
                    tflite.FullyConnectedOptionsWeightsFormat.DEFAULT)
                {
                    throw new NotSupportedException();
                }

                fusedActivationFunction = options.FusedActivationFunction;

                var perm = GetPerm(op, 1);
                rhs = Transpose(rhs, perm);
            }
            else
            {
                var batchMatMulOptions = op.BuiltinOptionsAsBatchMatMulOptions();
                if (batchMatMulOptions.AdjX)
                {
                    var perm = GetPerm(op, 0);
                    lhs = Transpose(lhs, perm);
                }

                if (batchMatMulOptions.AdjY)
                {
                    var perm = GetPerm(op, 1);
                    rhs = Transpose(rhs, perm);
                }
            }

            var bias = op.InputsLength == 3 && op.Inputs(2) != -1
                ? GetInputExprs(op, 2)
                : Expand(Cast(0, GetDataType(GetInputTensor(op, 0).Type)), new[] { otherTensor.Shape(0) }).Evaluate().AsTensor();

            var matmul = MatMul(lhs, rhs);
            List<string> outputNames = new() { GetOutputTensor(op, 0).Name + "_matmul" };
            matmul.Metadata.OutputNames = outputNames;
            outputNames.Clear();
            outputNames.Add(GetOutputTensor(op, 0).Name + "_bias");
            bias.Metadata.OutputNames = outputNames;
            var mm = matmul + bias;
            outputNames.Clear();
            outputNames.Add(GetOutputTensor(op, 0).Name);
            mm.Metadata.OutputNames = outputNames;

            return fusedActivationFunction switch
            {
                ActivationFunctionType.NONE => mm,
                ActivationFunctionType.RELU => F.NN.Relu(mm),
                ActivationFunctionType.RELU6 => F.NN.Relu6(mm),
                ActivationFunctionType.TANH => F.Math.Tanh(mm),
                _ => throw new NotImplementedException("Not supported FusedActivationFunction"),
            };
        }

        private int[] GetPerm(tflite.Operator op, int index)
        {
            var r = GetShapeArray(GetInputTensor(op, index)).Length;
            var perm = Enumerable.Range(0, r).ToArray();
            var tmp = perm[^1];
            perm[^1] = perm[^2];
            perm[^2] = tmp;
            return perm;
        }
    }
}
