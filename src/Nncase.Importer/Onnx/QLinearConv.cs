// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.IO;
using System.Linq;
using Nncase.IR;
using Onnx;
using static Nncase.IR.F.Math;
using static Nncase.IR.F.NN;
using static Nncase.IR.F.Tensors;
using F = Nncase.IR.F;

namespace Nncase.Importer
{
    public partial class OnnxImporter
    {
        private Expr VisitQLinearConv(in NodeProto op)
        {
            var (input, weights) = GetInputExprs<Expr, Expr>(op, 0, 3);
            var xScale = GetInputExpr<Expr>(op, 1);
            var xZeroPoint = GetInputExpr<Expr>(op, 2);
            var wScale = GetInputExpr<Expr>(op, 4);
            var wZeroPoint = GetInputExpr<Expr>(op, 5);
            var yScale = GetInputExpr<Expr>(op, 6);
            var yZeroPoint = GetInputExpr<Expr>(op, 7);
            var bias = op.Input.Count == 9 ? GetInputExpr<Expr>(op, 8) : null;
            var autoPad = GetStringAttribute(op, "auto_pad", "NOTSET");
            var dilation = GetDilationsAttribute(op);
            var group = GetIntAttribute(op, "group", 1);
            var strides = GetStrideAttribute(op);

            int stridesValueLen = (int)((TensorConst)strides).CheckedShape[0].FixedValue;
            for (var i = 0; i < stridesValueLen; i++)
            {
                System.Diagnostics.Trace.Assert(((TensorConst)strides).Value.Cast<long>()[i] <= (long)int.MaxValue);
            }

            int dilationValueLen = (int)((TensorConst)dilation).CheckedShape[0].FixedValue;
            for (var i = 0; i < dilationValueLen; i++)
            {
                System.Diagnostics.Trace.Assert(((TensorConst)dilation).Value.Cast<long>()[i] <= (long)int.MaxValue);
            }

            var pads = AutoPad(op, autoPad, input, weights, strides.ToArray<long>(), dilation);
            int[] strideArr = new int[stridesValueLen];
            for (var i = 0; i < stridesValueLen; i++)
            {
                strideArr[i] = ((TensorConst)strides).Value.Cast<int>()[i];
            }

            int[] dilationArr = new int[dilationValueLen];
            for (var i = 0; i < dilationValueLen; i++)
            {
                dilationArr[i] = ((TensorConst)dilation).Value.Cast<int>()[i];
            }

            var inputDeq = Dequantize(input, new QuantParam(((TensorConst)xZeroPoint).Value.ToScalar<int>(), ((TensorConst)xScale).Value.ToScalar<float>()), DataTypes.Float32);
            var weightsDeq = Dequantize(weights, new QuantParam(((TensorConst)wZeroPoint).Value.ToScalar<int>(), ((TensorConst)wScale).Value.ToScalar<float>()), DataTypes.Float32);

            if (bias == null)
            {
                int? ocNumber = (int)((TensorConst)weights).CheckedShape[0].FixedValue;
                var zeroBias = new TensorConst(new int[ocNumber == null ? default(int) : ocNumber.Value]);
                var conv = F.NN.Conv2D(inputDeq, weightsDeq, zeroBias, strideArr, pads, dilationArr, PadMode.Constant, group);
                return Quantize(conv, new QuantParam(((TensorConst)yZeroPoint).Value.ToScalar<int>(), ((TensorConst)yScale).Value.ToScalar<float>()), ((TensorConst)yZeroPoint).CheckedDataType);
            }
            else
            {
                var biasDeq = Dequantize(bias, new QuantParam(0, ((TensorConst)xScale).Value.ToScalar<float>() * ((TensorConst)wScale).Value.ToScalar<float>()), DataTypes.Float32);
                var conv = F.NN.Conv2D(inputDeq, weightsDeq, biasDeq, strideArr, pads, dilationArr, PadMode.Constant, group);
                return Quantize(conv, new QuantParam(((TensorConst)yZeroPoint).Value.ToScalar<int>(), ((TensorConst)yScale).Value.ToScalar<float>()), ((TensorConst)yZeroPoint).CheckedDataType);
            }
        }
    }
}
