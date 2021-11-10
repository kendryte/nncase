// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.IO;
using System.Linq;
using Nncase.IR;
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
            var stride = Const.FromSpan<int>(new[] {strideH, strideW}, new[] {2});
            var dilation = Const.FromSpan<int>(new[] {dilationH, dilationW}, new[] {2});
            var padding = Util.ConcatPadding(padH, padW);
            var clamp = ToFloatClampRange(options.FusedActivationFunction);
            return Util.NCHWToNHWC(F.Math.Clamp(
                F.NN.Conv2D(Util.NHWCToNCHW(input), Util.NHWCToNCHW(weights), bias, padding, stride, dilation,
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
            var stride = Const.FromSpan<int>(new[] {strideH, strideW}, new[] {2});
            var dilation = Const.FromSpan<int>(new[] {dilationH, dilationW}, new[] {2});
            var padding = Util.ConcatPadding(padH, padW);
            var depthMul = options.DepthMultiplier;
            if (depthMul != 1)
            {
                throw new NotSupportedException("DepthwiseConv2D with depth_multiplier:" + depthMul +
                                                " is not supported");
            }

            var clamp = ToFloatClampRange(options.FusedActivationFunction);
            return Util.NCHWToNHWC(F.Math.Clamp(
                F.NN.Conv2D(Util.NHWCToNCHW(input), Util.NHWCToNCHW(weights), bias, padding, stride, dilation,
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

        private static Expr GetWindowedOutputSize(Expr size, Expr filter, Expr stride, Expr dilation, Expr same, Expr ceilMode)
        {
            var effectiveFilterSize = ((filter - 1) * dilation) + 1;
            var trueBranch = ((size + stride - 1) / stride);
            var falseBranch = new If(ceilMode,
                ((size - effectiveFilterSize + stride) / stride),
                F.Math.Ceil((size - effectiveFilterSize + stride) / stride));
            return new If(same, trueBranch, falseBranch);
        }

        private static Expr[] GetWindowedPaddingValue(Expr inputSize, Expr outputSize, Expr filter, Expr stride, Expr dilation)
        {
            var effectiveFilterSize = ((filter - 1) * dilation) + 1;
            var padding = F.Math.Max(0, ((outputSize - 1) * stride) + effectiveFilterSize - inputSize);
            return new[] { padding / 2, padding - (padding / 2) };
        }
        
        private static Expr[] GetWindowedPadding(Expr inputSize, Expr filter, Expr stride, Expr dilation, Expr same)
        {
            var outputSize = GetWindowedOutputSize(inputSize, filter, stride, dilation, same, false);
            return GetWindowedPaddingValue(inputSize, outputSize, filter, stride, dilation);
        }

        // private static int GetWindowedOutputSize(int size, int filter, int stride, int dilation, bool same,
        //     bool ceilMode = false)
        // {
        //     var effectiveFilterSize = ((filter - 1) * dilation) + 1;
        //     if (same)
        //     {
        //         return (int)(((uint)size + stride - 1) / stride);
        //     }
        //     else
        //     {
        //         if (!ceilMode)
        //         {
        //             return (int)(((uint)size - effectiveFilterSize + stride) / stride);
        //         }
        //         else
        //         {
        //             return (int)Math.Ceiling((float)((uint)size - effectiveFilterSize + stride) / stride);
        //         }
        //     }
        // }
        //
        // private static int[] GetWindowedPadding(int inputSize, int filter, int stride, int dilation, bool same)
        // {
        //     var outputSize = GetWindowedOutputSize(inputSize, filter, stride, dilation, same);
        //     return GetWindowedPadding(inputSize, outputSize, filter, stride, dilation);
        // }
        //
        // private static int[] GetWindowedPadding(int inputSize, int outputSize, int filter, int stride, int dilation)
        // {
        //     var effectiveFilterSize = ((filter - 1) * dilation) + 1;
        //     var padding = Math.Max(0, ((outputSize - 1) * stride) + effectiveFilterSize - inputSize);
        //     return new[] { padding / 2, padding - (padding / 2) };
        // }
    }
}