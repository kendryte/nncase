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
            var padH = GetWindowedPadding(inH, fH, strideH, dilationH, options.Padding == tflite.Padding.SAME);
            var padW = GetWindowedPadding(inW, fW, strideW, dilationW, options.Padding == tflite.Padding.SAME);
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
            var padH = GetWindowedPadding(inH, fH, strideH, dilationH, options.Padding == tflite.Padding.SAME);
            var padW = GetWindowedPadding(inW, fW, strideW, dilationW, options.Padding == tflite.Padding.SAME);
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

        private static Expr GetWindowedOutputSize(Expr size, Expr filter, Expr stride, Expr dilation, bool same, bool ceilMode)
        {
            var effectiveFilterSize = ((filter - 1) * dilation) + 1;
            var falseBranch = ceilMode
                ? ((size - effectiveFilterSize + stride) / stride)
                : F.Tensors.Cast(F.Math.Ceil(
                    F.Tensors.Cast((size - effectiveFilterSize + stride), DataType.Float32) / 
                    F.Tensors.Cast(stride, DataType.Float32)),
                    DataType.Int32);
            var trueBranch = (size + stride - 1) / stride;
            return same ? trueBranch : falseBranch;
        }

        private static Expr[] GetWindowedPaddingValue(Expr inputSize, Expr outputSize, Expr filter, Expr stride, Expr dilation)
        {
            var effectiveFilterSize = ((filter - 1) * dilation) + 1;
            var padding = F.Math.Max(0, ((outputSize - 1) * stride) + effectiveFilterSize - inputSize);
            return new[] { F.Tensors.Cast(padding / 2, DataType.Int32), F.Tensors.Cast(padding - (padding / 2), DataType.Int32) };
        }

        public static Expr[] GetWindowedPadding(Expr inputSize, Expr filter, Expr stride, Expr dilation, bool same)
        {
            var outputSize = GetWindowedOutputSize(inputSize, filter, stride, dilation, same, false);
            return GetWindowedPaddingValue(inputSize, outputSize, filter, stride, dilation);
        }
    }
}