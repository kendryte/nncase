// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.IR;
using Onnx;
using static Nncase.IR.F.Math;
using static Nncase.IR.F.NN;
using F = Nncase.IR.F;

namespace Nncase.Importer
{
    public partial class OnnxImporter
    {
        private Expr VisitQLinearMatMul(in NodeProto op)
        {
            var (input_a, input_b) = GetInputExprs(op, 0, 3);
            var aScale = GetInputExpr(op, 1);
            var aZeroPoint = GetInputExpr(op, 2);
            var bScale = GetInputExpr(op, 4);
            var bZeroPoint = GetInputExpr(op, 5);
            var yScale = GetInputExpr(op, 6);
            var yZeroPoint = GetInputExpr(op, 7);

            var aDeq = Dequantize(input_a, new QuantParam(((TensorConst)aZeroPoint).Value.ToScalar<int>(), ((TensorConst)aScale).Value.ToScalar<float>()), DataTypes.Float32);
            var bDeq = Dequantize(input_b, new QuantParam(((TensorConst)bZeroPoint).Value.ToScalar<int>(), ((TensorConst)bScale).Value.ToScalar<float>()), DataTypes.Float32);
            var matmul = F.Tensors.MatMul(aDeq, bDeq);
            return Quantize(matmul, new QuantParam(((TensorConst)yZeroPoint).Value.ToScalar<int>(), ((TensorConst)yScale).Value.ToScalar<float>()), ((TensorConst)yZeroPoint).CheckedDataType);
        }
    }
}
