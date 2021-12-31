// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.IO;
using System.Linq;
using Nncase.IR;
using Nncase.IR.Tensors;
using F = Nncase.IR.F;
using TensorType = tflite.TensorType;
using Tuple = Nncase.IR.Tuple;

namespace Nncase.Importer.TFLite
{
    public partial class TFLiteImporter
    {
        private Expr VisitConv2D(in tflite.Operator op)
        {
            var (input, weights) = GetInputExprs(op, 0, 1);
            var bias = GetInputExprs(op, 2);
            var options = op.BuiltinOptionsAsConv2DOptions();
            var (inH, inW) = Util.GetHW(input);
            var (fH, fW) = Util.GetHW(weights);
            var strideH = options.StrideH;
            var strideW = options.StrideW;
            var dilationH = options.DilationHFactor;
            var dilationW = options.DilationWFactor;
            var padH = Util.GetWindowedPadding(inH, fH, strideH, dilationH, options.Padding == tflite.Padding.SAME);
            var padW = Util.GetWindowedPadding(inW, fW, strideW, dilationW, options.Padding == tflite.Padding.SAME);
            var stride = Const.FromSpan<int>(new[] { strideH, strideW }, new[] { 2 });
            var dilation = Const.FromSpan<int>(new[] { dilationH, dilationW }, new[] { 2 });
            var padding = Util.ConcatPadding(padH, padW);
            var clamp = ToFloatClampRange(options.FusedActivationFunction);
            return F.Tensors.NCHWToNHWC(F.Math.Clamp(
                F.NN.Conv2D(F.Tensors.NHWCToNCHW(input), F.Tensors.NHWCToNCHW(weights), bias, stride, padding, dilation,
                    PadMode.Constant, 1),
                clamp.Min, clamp.Max));
        }

        private Expr VisitDepthwiseConv2D(in tflite.Operator op)
        {
            var (input, weights) = GetInputExprs(op, 0, 1);
            var bias = GetInputExprs(op, 2);
            var options = op.BuiltinOptionsAsDepthwiseConv2DOptions();
            var (inH, inW) = Util.GetHW(input);
            var (fH, fW) = Util.GetHW(weights);
            var strideH = options.StrideH;
            var strideW = options.StrideW;
            var dilationH = options.DilationHFactor;
            var dilationW = options.DilationWFactor;
            var padH = Util.GetWindowedPadding(inH, fH, strideH, dilationH, options.Padding == tflite.Padding.SAME);
            var padW = Util.GetWindowedPadding(inW, fW, strideW, dilationW, options.Padding == tflite.Padding.SAME);
            var stride = Const.FromSpan<int>(new[] { strideH, strideW }, new[] { 2 });
            var dilation = Const.FromSpan<int>(new[] { dilationH, dilationW }, new[] { 2 });
            var padding = Util.ConcatPadding(padH, padW);
            var depthMul = options.DepthMultiplier;
            if (depthMul != 1)
            {
                throw new NotSupportedException("DepthwiseConv2D with depth_multiplier:" + depthMul +
                                                " is not supported");
            }

            var clamp = ToFloatClampRange(options.FusedActivationFunction);
            return F.Tensors.NCHWToNHWC(F.Math.Clamp(
                F.NN.Conv2D(F.Tensors.NHWCToNCHW(input), F.Tensors.NHWCToNCHW(weights), bias, stride, padding, dilation,
                    PadMode.Constant, 1),
                clamp.Min, clamp.Max));
        }

        private static ValueRange<float> ToFloatClampRange(tflite.ActivationFunctionType func) => func switch
        {
            tflite.ActivationFunctionType.NONE => ValueRange<float>.Full,
            tflite.ActivationFunctionType.RELU => (0f, float.PositiveInfinity),
            tflite.ActivationFunctionType.RELU_N1_TO_1 => (-1f, 1f),
            tflite.ActivationFunctionType.RELU6 => (0f, 6f),
            _ => throw new NotSupportedException("Unsupported Activation:" + func),
        };
    }
}