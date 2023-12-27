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
            var (input, weights) = GetInputExprs(op, 0, 3);
            var xScale = GetInputExpr(op, 1);
            var xZeroPoint = GetInputExpr(op, 2);
            var wScale = GetInputExpr(op, 4);
            var wZeroPoint = GetInputExpr(op, 5);
            var yScale = GetInputExpr(op, 6);
            var yZeroPoint = GetInputExpr(op, 7);
            var bias = op.Input.Count == 9 ? GetInputExpr(op, 8) : null;
            var autoPad = GetStringAttribute(op, "auto_pad", "NOTSET");
            var dilation = GetDilationsAttribute(op);
            var group = GetIntAttribute(op, "group", 1);
            var strides = GetStrideAttribute(op);

            int? stridesValueLen = ((TensorConst)strides).CheckedShape[0].Value;
            for (var i = 0; i < stridesValueLen; i++)
            {
                System.Diagnostics.Trace.Assert(((TensorConst)strides).Value.Cast<long>()[i] <= (long)int.MaxValue);
            }

            int? dilationValueLen = ((TensorConst)dilation).CheckedShape[0].Value;
            for (var i = 0; i < dilationValueLen; i++)
            {
                System.Diagnostics.Trace.Assert(((TensorConst)dilation).Value.Cast<long>()[i] <= (long)int.MaxValue);
            }

            var pads = AutoPad(op, autoPad, input, weights, strides.ToArray<long>(), dilation);
            int[] strideArr = new int[stridesValueLen == null ? default : stridesValueLen.Value];
            for (var i = 0; i < stridesValueLen; i++)
            {
                strideArr[i] = ((TensorConst)strides).Value.Cast<int>()[i];
            }

            var strideConst = new TensorConst(Tensor.From<int>(strideArr));

            int[] dilationArr = new int[dilationValueLen == null ? default : dilationValueLen.Value];
            for (var i = 0; i < dilationValueLen; i++)
            {
                dilationArr[i] = ((TensorConst)dilation).Value.Cast<int>()[i];
            }

            var dilationConst = new TensorConst(Tensor.From<int>(dilationArr));

            var inputDeq = Dequantize(input, new QuantParam(((TensorConst)xZeroPoint).Value.ToScalar<int>(), ((TensorConst)xScale).Value.ToScalar<float>()), DataTypes.Float32);
            var weightsDeq = Dequantize(weights, new QuantParam(((TensorConst)wZeroPoint).Value.ToScalar<int>(), ((TensorConst)wScale).Value.ToScalar<float>()), DataTypes.Float32);

            if (bias == null)
            {
                int? ocNumber = ((TensorConst)weights).CheckedShape[0].Value;
                var zeroBias = new TensorConst(new int[ocNumber == null ? default(int) : ocNumber.Value]);
                var conv = F.NN.Conv2D(inputDeq, weightsDeq, zeroBias, strideConst, pads, dilationConst, PadMode.Constant, group);
                return Quantize(conv, new QuantParam(((TensorConst)yZeroPoint).Value.ToScalar<int>(), ((TensorConst)yScale).Value.ToScalar<float>()), ((TensorConst)yZeroPoint).CheckedDataType);
            }
            else
            {
                var biasDeq = Dequantize(bias, new QuantParam(0, ((TensorConst)xScale).Value.ToScalar<float>() * ((TensorConst)wScale).Value.ToScalar<float>()), DataTypes.Float32);
                var conv = F.NN.Conv2D(inputDeq, weightsDeq, biasDeq, strideConst, pads, dilationConst, PadMode.Constant, group);
                return Quantize(conv, new QuantParam(((TensorConst)yZeroPoint).Value.ToScalar<int>(), ((TensorConst)yScale).Value.ToScalar<float>()), ((TensorConst)yZeroPoint).CheckedDataType);
            }
        }
    }
}
