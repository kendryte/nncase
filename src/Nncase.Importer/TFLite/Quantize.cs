// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.IO;
using Nncase.IR;
using F = Nncase.IR.F;

namespace Nncase.Importer.TFLite
{
    public partial class TFLiteImporter
    {
        private Expr VisitQuantize(in tflite.Operator op)
        {
            var input = GetInputExprs(op, 0);
            var outputTensor = GetOutputTensor(op, 0);
            var param = outputTensor.Quantization ?? throw new InvalidDataException(
                "Quantize Parameter not found in tflite Quantize importer");
            var quantParam = new Tuple((Const)param.ZeroPoint(0), (Const)param.Scale(0));
            return F.Tensors.Quantize(input, quantParam, GetDataType(outputTensor.Type));
        }

        private Expr VisitDeQuantize(in tflite.Operator op)
        {
            var input = GetInputExprs(op, 0);
            var outputTensor = GetOutputTensor(op, 0);
            var param = outputTensor.Quantization ?? throw new InvalidDataException(
                "Quantize Parameter not found in tflite DeQuantize importer");
            var deqParam = new Tuple((Const)param.ZeroPoint(0), (Const)param.Scale(0));
            return F.Tensors.DeQuantize(input, deqParam, GetDataType(outputTensor.Type));
        }
    }
}