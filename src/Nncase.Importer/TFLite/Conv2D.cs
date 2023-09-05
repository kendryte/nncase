// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using Nncase.IR;
using static Nncase.IR.F.Math;
using static Nncase.IR.F.NN;
using static Nncase.IR.F.Tensors;
using F = Nncase.IR.F;

namespace Nncase.Importer.TFLite
{
    public partial class TFLiteImporter
    {
        private static ValueRange<float> ToFloatClampRange(tflite.ActivationFunctionType func) => func switch
        {
            tflite.ActivationFunctionType.NONE => ValueRange<float>.Full,
            tflite.ActivationFunctionType.RELU => (0f, float.PositiveInfinity),
            tflite.ActivationFunctionType.RELU_N1_TO_1 => (-1f, 1f),
            tflite.ActivationFunctionType.RELU6 => (0f, 6f),
            _ => throw new NotSupportedException("Unsupported Activation:" + func),
        };

        private Expr VisitConv2D(in tflite.Operator op)
        {
            var (input, weights) = GetInputExprs(op, 0, 1);
            input = F.Tensors.NHWCToNCHW(input);
            weights = F.Tensors.NHWCToNCHW(weights);
            var bias = GetInputExprs(op, 2);
            var options = op.BuiltinOptionsAsConv2DOptions();
            var strideH = options.StrideH;
            var strideW = options.StrideW;
            var dilationH = options.DilationHFactor;
            var dilationW = options.DilationWFactor;
            var stride = Tensor.From<int>(new[] { strideH, strideW }, new[] { 2 });
            var dilation = Tensor.From<int>(new[] { dilationH, dilationW }, new[] { 2 });
            var padding = Util.GetPaddings(input, weights, stride,
                dilation, options.Padding == tflite.Padding.SAME, false);

            var clamp = ToFloatClampRange(options.FusedActivationFunction);

            var inputQuantParams = GetInputQuantParams(op, 0);
            var weightsQuantParams = GetInputQuantParams(op, 1);
            var biasQuantParams = GetInputQuantParams(op, 2);
            var outputQuantParams = GetOutputQuantParams(op, 0);

            if (inputQuantParams != null)
            {
                input = Dequantize(input, new QuantParam(inputQuantParams[0].ZeroPoint, inputQuantParams[0].Scale), DataTypes.Float32);
            }

            if (weightsQuantParams != null)
            {
                weights = Dequantize(weights, new QuantParam(weightsQuantParams[0].ZeroPoint, weightsQuantParams[0].Scale), DataTypes.Float32);
            }

            if (biasQuantParams != null)
            {
                bias = Dequantize(bias, new QuantParam(biasQuantParams[0].ZeroPoint, biasQuantParams[0].Scale), DataTypes.Float32);
            }

            if (outputQuantParams != null)
            {
                if (GetOutputTensor(op, 0).Type == tflite.TensorType.INT8)
                {
                    return F.Tensors.NCHWToNHWC(Quantize(
                        F.NN.Conv2D(
                            input,
                            weights,
                            bias,
                            stride,
                            padding,
                            dilation,
                            PadMode.Constant,
                            1,
                            new[] { clamp.Min, clamp.Max }),
                        new QuantParam(outputQuantParams[0].ZeroPoint, outputQuantParams[0].Scale),
                        DataTypes.Int8));
                }
                else if (GetOutputTensor(op, 0).Type == tflite.TensorType.UINT8)
                {
                    return F.Tensors.NCHWToNHWC(Quantize(
                        F.NN.Conv2D(
                            input,
                            weights,
                            bias,
                            stride,
                            padding,
                            dilation,
                            PadMode.Constant,
                            1,
                            new[] { clamp.Min, clamp.Max }),
                        new QuantParam(outputQuantParams[0].ZeroPoint, outputQuantParams[0].Scale),
                        DataTypes.UInt8));
                }
                else
                {
                    throw new NotSupportedException("Unsupported qat quant type");
                }
            }
            else
            {
                return F.Tensors.NCHWToNHWC(
                    F.NN.Conv2D(
                        input,
                        weights,
                        bias,
                        stride,
                        padding,
                        dilation,
                        PadMode.Constant,
                        1,
                        new[] { clamp.Min, clamp.Max }));
            }
        }

        private Expr VisitDepthwiseConv2D(in tflite.Operator op)
        {
            var (input, weights) = GetInputExprs(op, 0, 1);
            var bias = GetInputExprs(op, 2);
            input = F.Tensors.NHWCToNCHW(input);
            weights = F.Tensors.Transpose(weights, new[] { 3, 0, 1, 2 });
            var options = op.BuiltinOptionsAsDepthwiseConv2DOptions();
            var strideH = options.StrideH;
            var strideW = options.StrideW;
            var dilationH = options.DilationHFactor;
            var dilationW = options.DilationWFactor;
            var stride = Tensor.From<int>(new[] { strideH, strideW }, new[] { 2 });
            var dilation = Tensor.From<int>(new[] { dilationH, dilationW }, new[] { 2 });
            var padding = Util.GetPaddings(input, weights, stride,
                dilation, options.Padding == tflite.Padding.SAME, false);

            var depthMul = options.DepthMultiplier;
            if (depthMul != 1)
            {
                throw new NotSupportedException("DepthwiseConv2D with depth_multiplier:" + depthMul +
                                                " is not supported");
            }

            var clamp = ToFloatClampRange(options.FusedActivationFunction);

            return F.Tensors.NCHWToNHWC(
                F.NN.Conv2D(
                    input,
                    weights,
                    bias,
                    stride,
                    padding,
                    dilation,
                    PadMode.Constant,
                    Util.ShapeIndex(weights, 0),
                    new[] { clamp.Min, clamp.Max }));
        }
    }
}
