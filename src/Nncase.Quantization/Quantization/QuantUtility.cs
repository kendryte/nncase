﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Data;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using Nncase.Evaluator;
using Nncase.IR;
using OrtKISharp;
using SMath = System.Math;

namespace Nncase.Quantization;

/// <summary>
/// QuantAlgorithm utility.
/// </summary>
public static class QuantAlgorithmUtility
{
    public static Tensor<float> SquantWeights(Tensor<float> inputWeights, Tensor<float> inputWeightsRanges, ReadOnlySpan<int> inputWeightsShape, QuantMode quantMode, int bits, bool isByChannel)
    {
        float qMax, qMin;
        if (quantMode == QuantMode.UnsignedMode)
        {
            qMax = (1 << bits) - 1;
            qMin = 0;
        }
        else if (quantMode == QuantMode.SignedAsymmetricMode)
        {
            qMax = (1 << (bits - 1)) - 1;
            qMin = -(1 << (bits - 1));
        }
        else
        {
            qMax = (1 << (bits - 1)) - 1;
            qMin = -(1 << (bits - 1)) + 1;
        }

        var inWShape = inputWeights.Shape.Select(x => (long)x.FixedValue).ToArray();
        OrtKISharp.Tensor x, delta, zeroPoint;
        if (inputWeightsShape.Length == 4)
        {
            var outChannel = inputWeightsShape[0];
            x = inputWeights.ToOrtTensor();

            if (isByChannel)
            {
                float[] deltaArr = new float[inputWeights.Length];
                float[] zeroPointArr = new float[inputWeights.Length];
                int eachChannelSize = inputWeights.Length / outChannel;

                Parallel.For(0, outChannel, c =>
                {
                    var xMin = inputWeightsRanges[c, 0];
                    var xMax = inputWeightsRanges[c, 1];
                    var deltaTmp = (xMax - xMin) / (qMax - qMin);
                    var zeroPointTmp = System.Math.Round(((xMax * qMin) - (xMin * qMax)) / (xMax - xMin));
                    for (int i = 0; i < eachChannelSize; i++)
                    {
                        deltaArr[(c * eachChannelSize) + i] = deltaTmp;
                        zeroPointArr[(c * eachChannelSize) + i] = (float)zeroPointTmp;
                    }
                });
                delta = OrtKISharp.Tensor.MakeTensor(deltaArr, inWShape);
                zeroPoint = OrtKISharp.Tensor.MakeTensor(zeroPointArr, inWShape);
            }
            else
            {
                throw new NotSupportedException("By tensor weights quant is not supported.");
            }
        }
        else
        {
            var outChannel = inputWeightsShape[0];
            var inChannel = inputWeightsShape[1];
            x = inputWeights.ToOrtTensor();

            if (isByChannel)
            {
                float[] deltaArr = new float[inputWeights.Length];
                float[] zeroPointArr = new float[inputWeights.Length];
                int eachChannelSize = inputWeights.Length / outChannel;

                Parallel.For(0, outChannel, c =>
                {
                    var xMin = inputWeightsRanges[c, 0];
                    var xMax = inputWeightsRanges[c, 1];
                    var deltaTmp = (xMax - xMin) / (qMax - qMin);
                    var zeroPointTmp = System.Math.Round(((xMax * qMin) - (xMin * qMax)) / (xMax - xMin));

                    for (int i = 0; i < eachChannelSize; i++)
                    {
                        deltaArr[(c * eachChannelSize) + i] = deltaTmp;
                        zeroPointArr[(c * eachChannelSize) + i] = (float)zeroPointTmp;
                    }
                });

                delta = OrtKISharp.Tensor.MakeTensor(deltaArr, inWShape);
                zeroPoint = OrtKISharp.Tensor.MakeTensor(zeroPointArr, inWShape);
            }
            else
            {
                throw new NotSupportedException("By layer weights quant is not supported.");
            }
        }

        var quantTensor = OrtKI.Add(OrtKI.Div(x, delta), zeroPoint);
        var xInt = AdaptiveRound(quantTensor, qMin, qMax); // SQuant量化
        var xQuant = OrtKI.Clip(xInt, OrtKISharp.Tensor.FromScalar<float>(qMin), OrtKISharp.Tensor.FromScalar<float>(qMax));
        var xDequant = (xQuant - zeroPoint) * delta;
        var res = Tensor.From<float>(xDequant.ToArray<float>(), inputWeights.Shape);
        quantTensor.Dispose();
        xQuant.Dispose();
        xDequant.Dispose();
        return res;
    }

    private static void RoundingForward(float roundingErrorSum, Span<float> roundingNumberMem, Span<float> roundingErrorMem, Span<float> numberMem, Span<float> errorMem, Span<float> priorityMem, Span<long> orderMem, Span<float> priority1Mem)
    {
        int topK = (int)System.Math.Round(System.Math.Abs(roundingErrorSum));
        bool overSquant = topK >= System.Math.Abs(roundingErrorSum);
        if (topK > 0)
        {
            var orderTmpArr = orderMem.Slice(0, topK);

            for (int i = 0; i < orderTmpArr.Length; i++)
            {
                var index = (int)orderTmpArr[i];
                roundingErrorMem[index] = errorMem[index];
                roundingNumberMem[index] = numberMem[index];
            }

            if (overSquant)
            {
                var index = (int)orderMem[topK - 1];
                priority1Mem[index] = System.Math.Abs(roundingErrorMem[index]);
            }
            else
            {
                var index = (int)orderMem[topK];
                priorityMem[index] = System.Math.Abs(roundingErrorMem[index]);
            }
        }
    }

    private static void SQuantFunc(OrtKISharp.Tensor roundingErrorSum, OrtKISharp.Tensor roundingNumber, OrtKISharp.Tensor roundingError, OrtKISharp.Tensor upNumber, OrtKISharp.Tensor upError, OrtKISharp.Tensor upPriority, OrtKISharp.Tensor upOrder, OrtKISharp.Tensor downNumber, OrtKISharp.Tensor downError, OrtKISharp.Tensor downPriority, OrtKISharp.Tensor downOrder, bool getNumberOnly)
    {
        var roundingNumberShape = roundingNumber.Shape.Select(x => (int)x).ToArray();
        if (roundingNumberShape.Length != 3)
        {
            throw new InvalidOperationException("Error");
        }

        var batches = roundingNumberShape[0];
        var inputChannel = roundingNumberShape[1];
        var sizePreChannel = roundingNumberShape[2];
        var roundingErrorSumArr = roundingErrorSum.ToArray<float>();
        var loopSize = (long)batches * inputChannel;
        Parallel.For(0, loopSize, currentIndex =>
        {
            var n = currentIndex / inputChannel;
            var c = currentIndex % inputChannel;
            using var starts = OrtKISharp.Tensor.MakeTensor(new long[] { n, c }, new long[] { 2 });
            using var ends = OrtKISharp.Tensor.MakeTensor(new long[] { n + 1, c + 1 }, new long[] { 2 });
            using var axes = OrtKISharp.Tensor.MakeTensor(new long[] { 0, 1 }, new long[] { 2 });
            using var steps = OrtKISharp.Tensor.MakeTensor(new long[] { 1, 1 }, new long[] { 2 });

            Span<float> Sl(OrtKISharp.Tensor tensor)
            {
                var span = MemoryMarshal.Cast<byte, float>(tensor.BytesBuffer);
                var begin = currentIndex * sizePreChannel;
                return span.Slice((int)begin, sizePreChannel);
            }

            Span<long> SlInt(OrtKISharp.Tensor tensor)
            {
                var span = MemoryMarshal.Cast<byte, long>(tensor.BytesBuffer);
                var begin = currentIndex * sizePreChannel;
                return span.Slice((int)begin, sizePreChannel);
            }

            var roundingNumberTmp = Sl(roundingNumber);
            var roundingErrorTmp = Sl(roundingError);

            var upNumberSlice = Sl(upNumber);
            var upErrorSlice = Sl(upError);
            var upOrderSlice = SlInt(upOrder);
            var downNumberSlice = Sl(downNumber);
            var downErrorSlice = Sl(downError);
            var downOrderSlice = SlInt(downOrder);

            Span<float> priorityTmp;
            Span<float> priority1Tmp;
            if (roundingErrorSumArr[currentIndex] < 0)
            {
                priorityTmp = Sl(upPriority);
                priority1Tmp = Sl(downPriority);
                RoundingForward(roundingErrorSumArr[currentIndex], roundingNumberTmp, roundingErrorTmp, upNumberSlice, upErrorSlice, priorityTmp, upOrderSlice, priority1Tmp);
            }
            else
            {
                priorityTmp = Sl(downPriority);
                priority1Tmp = Sl(upPriority);
                RoundingForward(roundingErrorSumArr[currentIndex], roundingNumberTmp, roundingErrorTmp, downNumberSlice, downErrorSlice, priorityTmp, downOrderSlice, priority1Tmp);
            }
        });
    }

    private static OrtKISharp.Tensor AdaptiveRound(OrtKISharp.Tensor x, float tMin, float tMax)
    {
        bool squantK = true;
        bool squantC = true;

        var roundingNumber = OrtKI.Round(x); // round取整值
        var roundingError = roundingNumber - x; // 误差
        var zeros = OrtKISharp.Tensor.MakeTensor(Enumerable.Repeat(0.0f, (int)roundingError.Length).ToArray(), roundingError.Shape);

        var upNumber = roundingNumber;
        var upError = roundingError;
        upError = OrtKI.Where(OrtKI.Greater(x, tMax), zeros, upError); // 边界上的值不能再调整，所以去除
        upError = OrtKI.Where(OrtKI.Greater(upError, 0.0f), zeros, upError); // 误差为正的都设为0，即up对应“原值>量化值”的集合
        var upPriority = OrtKI.Abs(upError);

        upError = OrtKI.Where(OrtKI.Not(OrtKI.Equal(upError, 0.0f)), upError + 1.0f, upError); // up集合中，Flip翻转后对应的误差
        upNumber = OrtKI.Where(OrtKI.Not(OrtKI.Equal(upError, 0.0f)), upNumber + 1.0f, upNumber); // up集合中，Flip翻转后对应的取整值

        var downNumber = roundingNumber;
        var downError = roundingError;
        downError = OrtKI.Where(OrtKI.Less(x, tMin), zeros, downError); // 边界上的值不能再调整，所以去除
        downError = OrtKI.Where(OrtKI.Less(downError, 0.0f), zeros, downError); // 误差为负的都设为0，即down对应“原值<量化值”的集合
        var downPriority = OrtKI.Abs(downError);

        downError = OrtKI.Where(OrtKI.Not(OrtKI.Equal(downError, 0.0f)), downError - 1.0f, downError); // down集合中，Flip翻转后对应的误差
        downNumber = OrtKI.Where(OrtKI.Not(OrtKI.Equal(downError, 0.0f)), downNumber - 1.0f, downNumber); // down集合中，Flip翻转后对应的取整值

        var xTmp = OrtKI.Reshape(x, new long[] { x.Shape[0], x.Shape[1], -1 }, 0);
        var converShape = xTmp.Shape; // HW维度合并
        if (converShape[2] == 1)
        {
            squantK = false; // 只有一个元素时， 不做K的逼近
        }

        if (squantK)
        {
            var roundingErrorSum = OrtKI.ReduceSum(OrtKI.Reshape(roundingError, converShape, 0), new long[] { -1 }, 0, 0);
            var reshapeUpPriority = OrtKI.Reshape(upPriority, converShape, 0);
            var upPriorityK = reshapeUpPriority.Shape[^1];
            var sortRet = OrtKI.TopK(reshapeUpPriority, OrtKISharp.Tensor.MakeTensor(new long[] { upPriorityK }, new long[] { 1 }), -1, 1, 1);
            var upOrder = sortRet[1];
            var reshapeDownPriority = OrtKI.Reshape(downPriority, converShape, 0);
            var downPriorityK = reshapeDownPriority.Shape[^1];
            sortRet = OrtKI.TopK(reshapeDownPriority, OrtKISharp.Tensor.MakeTensor(new long[] { downPriorityK }, new long[] { 1 }), -1, 1, 1);
            var downOrder = sortRet[1];
            upPriority *= 0.0f;
            downPriority *= 0.0f;

            roundingNumber = OrtKI.Reshape(roundingNumber, converShape, 0);
            roundingError = OrtKI.Reshape(roundingError, converShape, 0);
            upNumber = OrtKI.Reshape(upNumber, converShape, 0);
            upError = OrtKI.Reshape(upError, converShape, 0);
            upPriority = OrtKI.Reshape(upPriority, converShape, 0);
            downNumber = OrtKI.Reshape(downNumber, converShape, 0);
            downError = OrtKI.Reshape(downError, converShape, 0);
            downPriority = OrtKI.Reshape(downPriority, converShape, 0);
            SQuantFunc(roundingErrorSum, roundingNumber, roundingError, upNumber, upError, upPriority, upOrder, downNumber, downError, downPriority, downOrder, false);
            roundingNumber = OrtKI.Reshape(roundingNumber, x.Shape, 0);
            roundingError = OrtKI.Reshape(roundingError, x.Shape, 0);
            upPriority = OrtKI.Reshape(upPriority, x.Shape, 0);
            downPriority = OrtKI.Reshape(downPriority, x.Shape, 0);
        }

        if (squantC)
        {
            converShape = new long[] { 1, x.Shape[0], -1 };
            var roundingErrorSum = OrtKI.ReduceSum(OrtKI.Reshape(roundingError, converShape, 0), new long[] { -1 }, 0, 0);
            var reshapePriority = OrtKI.Reshape(upPriority, converShape, 0);
            var upPriorityK = reshapePriority.Shape[^1];
            var sortRet = OrtKI.TopK(reshapePriority, OrtKISharp.Tensor.MakeTensor(new long[] { upPriorityK }, new long[] { 1 }), -1, 1, 1);
            var upOrder = sortRet[1];
            var reshapeDownPriority = OrtKI.Reshape(downPriority, converShape, 0);
            var downPriorityK = reshapeDownPriority.Shape[^1];
            sortRet = OrtKI.TopK(reshapeDownPriority, OrtKISharp.Tensor.MakeTensor(new long[] { downPriorityK }, new long[] { 1 }), -1, 1, 1);
            var downOrder = sortRet[1];

            roundingNumber = OrtKI.Reshape(roundingNumber, converShape, 0);
            roundingError = OrtKI.Reshape(roundingError, converShape, 0);
            upNumber = OrtKI.Reshape(upNumber, converShape, 0);
            upError = OrtKI.Reshape(upError, converShape, 0);
            upPriority = OrtKI.Reshape(upPriority, converShape, 0);
            downNumber = OrtKI.Reshape(downNumber, converShape, 0);
            downError = OrtKI.Reshape(downError, converShape, 0);
            downPriority = OrtKI.Reshape(downPriority, converShape, 0);
            SQuantFunc(roundingErrorSum, roundingNumber, roundingError, upNumber, upError, upPriority, upOrder, downNumber, downError, downPriority, downOrder, true);
        }

        roundingNumber = OrtKI.Reshape(roundingNumber, x.Shape, 0);

        return roundingNumber!;
    }
}
