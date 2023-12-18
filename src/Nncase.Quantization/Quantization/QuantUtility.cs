// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Data;
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

        OrtKISharp.Tensor x, delta, zeroPoint;
        if (inputWeightsShape.Length == 4)
        {
            var outChannel = inputWeightsShape[0];
            var inChannel = inputWeightsShape[1];
            var filterH = inputWeightsShape[2];
            var filterW = inputWeightsShape[3];
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
                delta = OrtKISharp.Tensor.MakeTensor(deltaArr, new long[] { outChannel, inChannel, filterH, filterW });
                zeroPoint = OrtKISharp.Tensor.MakeTensor(zeroPointArr, new long[] { outChannel, inChannel, filterH, filterW });
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

                delta = OrtKISharp.Tensor.MakeTensor(deltaArr, new long[] { outChannel, inChannel });
                zeroPoint = OrtKISharp.Tensor.MakeTensor(zeroPointArr, new long[] { outChannel, inChannel });
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
        // quantTensor.Dispose();
        // xQuant.Dispose();
        // GC.Collect();
        var res = Tensor.From<float>(xDequant.ToArray<float>(), inputWeights.Shape);
        // xDequant.Dispose();
        return res;
    }

    private static void RoundingForward(float roundingErrorSum, OrtKISharp.Tensor roundingNumber, OrtKISharp.Tensor roundingError, OrtKISharp.Tensor number, OrtKISharp.Tensor error, OrtKISharp.Tensor priority, OrtKISharp.Tensor order, OrtKISharp.Tensor priority1)
    {
        var roundingNumberMem = roundingNumber.ToArray<float>();
        var roundingErrorMem = roundingError.ToArray<float>();
        var priorityMem = priority.ToArray<float>();
        var priority1Mem = priority1.ToArray<float>();
        int topK = (int)System.Math.Round(System.Math.Abs(roundingErrorSum));
        bool overSquant = topK >= System.Math.Abs(roundingErrorSum);
        if (topK > 0)
        {
            var starts = OrtKISharp.Tensor.MakeTensor(new long[] { 0 }, new long[] { 1 });
            var ends = OrtKISharp.Tensor.MakeTensor(new long[] { topK }, new long[] { 1 });
            var axes = OrtKISharp.Tensor.MakeTensor(new long[] { 0 }, new long[] { 1 });
            var steps = OrtKISharp.Tensor.MakeTensor(new long[] { 1 }, new long[] { 1 });

            var orderTmp = OrtKI.Slice(order, starts, ends, axes, steps);

            var orderTmpArr = MemoryMarshal.Cast<byte, long>(orderTmp.BytesBuffer);
            var orderArr = MemoryMarshal.Cast<byte, int>(order.BytesBuffer);
            var errorArr = MemoryMarshal.Cast<byte, float>(error.BytesBuffer);
            var numberArr = MemoryMarshal.Cast<byte, float>(number.BytesBuffer);
            for (int i = 0; i < orderTmp.Length; i++)
            {
                var index = (int)orderTmpArr[i];
                roundingErrorMem[index] = errorArr[index];
                roundingNumberMem[index] = numberArr[index];
            }

            if (overSquant)
            {
                var index = orderArr[topK - 1];
                priority1Mem[index] = System.Math.Abs(roundingErrorMem[index]);
            }
            else
            {
                var index = orderArr[topK];
                priorityMem[index] = System.Math.Abs(roundingErrorMem[index]);
            }
            orderTmp.Dispose();
        }
    }

    private static void SQuantFunc(OrtKISharp.Tensor roundingErrorSum, OrtKISharp.Tensor roundingNumber,
        OrtKISharp.Tensor roundingError, OrtKISharp.Tensor upNumber, OrtKISharp.Tensor upError,
        OrtKISharp.Tensor upPriority, OrtKISharp.Tensor upOrder, OrtKISharp.Tensor downNumber,
        OrtKISharp.Tensor downError, OrtKISharp.Tensor downPriority, OrtKISharp.Tensor downOrder, bool getNumberOnly)
    {
        var roundingNumberShape = roundingNumber.Shape.Select(x => (int)x).ToArray();
        var batches = roundingNumberShape[0];
        var inputChannel = roundingNumberShape[1];
        int totalSize = 1;
        for (int i = 0; i < roundingNumberShape.Length; i++)
        {
            totalSize *= roundingNumberShape[i];
        }

        var oneBatchSize = totalSize / batches;
        var oneInputChannelSize = oneBatchSize / inputChannel;

        var roundingErrorSumArr = roundingErrorSum.ToArray<float>();

        Parallel.For(0, (long)batches * inputChannel, currentIndex =>
            // for (int currentIndex = 0; currentIndex < batches * inputChannel; currentIndex++)
        {
            // Console.WriteLine("For Index");
            // Console.WriteLine(currentIndex);
            var roundingNumberMem = MemoryMarshal.Cast<byte, float>(roundingNumber.BytesBuffer);
            var roundingErrorMem = MemoryMarshal.Cast<byte, float>(roundingError.BytesBuffer);
            var upPriorityMem = MemoryMarshal.Cast<byte, float>(upPriority.BytesBuffer);
            var downPriorityMem = MemoryMarshal.Cast<byte, float>(downPriority.BytesBuffer);

            System.DateTime start, end;
            start = System.DateTime.Now;
            var n = currentIndex / inputChannel;
            var c = currentIndex % inputChannel;
            var starts = OrtKISharp.Tensor.MakeTensor(new long[] { n, c }, new long[] { 2 });
            var ends = OrtKISharp.Tensor.MakeTensor(new long[] { n + 1, c + 1 }, new long[] { 2 });
            var axes = OrtKISharp.Tensor.MakeTensor(new long[] { 0, 1 }, new long[] { 2 });
            var steps = OrtKISharp.Tensor.MakeTensor(new long[] { 1, 1 }, new long[] { 2 });

            OrtKISharp.Tensor Sl(OrtKISharp.Tensor tensor)
            {
                var s = OrtKI.Slice(tensor, starts, ends, axes, steps);
                var ret = OrtKI.Squeeze(s, axes);
                // 这里dispose s会引起原始的挂掉
                s.Dispose();
                return s;
                // return IR.F.Tensors.Squeeze(IR.F.Tensors.Slice(tensor, starts, ends, axes, steps), axes);
            }

            OrtKISharp.Tensor tmp;
            tmp = OrtKI.Slice(roundingNumber, starts, ends, axes, steps);
            var roundingNumberTmp = OrtKI.Squeeze(tmp, axes);
            tmp.Dispose();
            tmp = OrtKI.Slice(roundingError, starts, ends, axes, steps);
            var roundingErrorTmp = OrtKI.Squeeze(tmp, axes);
            tmp.Dispose();
            tmp = OrtKI.Slice(upNumber, starts, ends, axes, steps);
            var upNumberSlice = OrtKI.Squeeze(tmp, axes);
            tmp.Dispose();
            tmp = OrtKI.Slice(upError, starts, ends, axes, steps);
            var upErrorSlice = OrtKI.Squeeze(tmp, axes);
            tmp.Dispose();
            tmp = OrtKI.Slice(upOrder, starts, ends, axes, steps);
            var upOrderSlice = OrtKI.Squeeze(tmp, axes);
            tmp.Dispose();
            tmp = OrtKI.Slice(downNumber, starts, ends, axes, steps);
            var downNumberSlice = OrtKI.Squeeze(tmp, axes);
            tmp.Dispose();
            tmp = OrtKI.Slice(downError, starts, ends, axes, steps);
            var downErrorSlice = OrtKI.Squeeze(tmp, axes);
            tmp.Dispose();
            tmp = OrtKI.Slice(downOrder, starts, ends, axes, steps);
            var downOrderSlice = OrtKI.Squeeze(tmp, axes);
            tmp.Dispose();

            var offset = (int)((n * oneBatchSize) + (c * oneInputChannelSize));
            OrtKISharp.Tensor priorityTmp;
            OrtKISharp.Tensor priority1Tmp;
            if (roundingErrorSumArr[currentIndex] < 0)
            {
                priorityTmp = OrtKI.Squeeze(OrtKI.Slice(upPriority, starts, ends, axes, steps), axes);
                priority1Tmp = OrtKI.Squeeze(OrtKI.Slice(downPriority, starts, ends, axes, steps), axes);
                RoundingForward(roundingErrorSumArr[currentIndex], roundingNumberTmp, roundingErrorTmp,
                    upNumberSlice, upErrorSlice, priorityTmp, upOrderSlice, priority1Tmp);

                // var roundingNumberTmpArr = roundingNumberTmp.ToArray<float>();
                // var roundingErrorTmpArr = roundingErrorTmp.ToArray<float>();
                // var priorityTmpArr = priorityTmp.ToArray<float>();
                // var priority1TmpArr = priority1Tmp.ToArray<float>();
                var roundingNumberTmpArr = roundingNumberTmp.ToArray<float>();
                var roundingErrorTmpArr = roundingErrorTmp.ToArray<float>();
                var priorityTmpArr = priorityTmp.ToArray<float>();
                var priority1TmpArr = priority1Tmp.ToArray<float>();
                for (int i = 0; i < roundingNumberTmp.Length; i++)
                {
                    roundingNumberMem[offset + i] =
                        roundingNumberTmpArr[i];
                }

                if (!getNumberOnly)
                {
                    for (int i = 0; i < roundingErrorTmp.Length; i++)
                    {
                        roundingErrorMem[offset + i] =
                            roundingErrorTmpArr[i];
                    }

                    for (int i = 0; i < priorityTmp.Length; i++)
                    {
                        upPriorityMem[offset + i] = priorityTmpArr[i];
                    }

                    for (int i = 0; i < priority1Tmp.Length; i++)
                    {
                        downPriorityMem[offset + i] = priority1TmpArr[i];
                    }
                }
            }
            else
            {
                priorityTmp = OrtKI.Squeeze(OrtKI.Slice(downPriority, starts, ends, axes, steps), axes);
                priority1Tmp = OrtKI.Squeeze(OrtKI.Slice(upPriority, starts, ends, axes, steps), axes);
                RoundingForward(roundingErrorSumArr[currentIndex], roundingNumberTmp, roundingErrorTmp,
                    downNumberSlice, downErrorSlice, priorityTmp, downOrderSlice, priority1Tmp);

                var roundingNumberTmpArr = roundingNumberTmp.ToArray<float>();
                var roundingErrorTmpArr = roundingErrorTmp.ToArray<float>();
                var priorityTmpArr = priorityTmp.ToArray<float>();
                var priority1TmpArr = priority1Tmp.ToArray<float>();

                for (int i = 0; i < roundingNumberTmp.Length; i++)
                {
                    roundingNumberMem[offset + i] =
                        roundingNumberTmpArr[i];
                }

                if (!getNumberOnly)
                {
                    for (int i = 0; i < roundingErrorTmp.Length; i++)
                    {
                        roundingErrorMem[offset + i] =
                            roundingErrorTmpArr[i];
                    }

                    for (int i = 0; i < priorityTmp.Length; i++)
                    {
                        downPriorityMem[offset + i] = priorityTmpArr[i];
                    }

                    for (int i = 0; i < priority1Tmp.Length; i++)
                    {
                        upPriorityMem[offset + i] = priority1TmpArr[i];
                    }
                }
            }

            priorityTmp.Dispose();
            priority1Tmp.Dispose();
            roundingNumberTmp.Dispose();
            roundingErrorTmp.Dispose();
            upNumberSlice.Dispose();
            upErrorSlice.Dispose();
            upOrderSlice.Dispose();
            downNumberSlice.Dispose();
            downErrorSlice.Dispose();
            downOrderSlice.Dispose();
            starts.Dispose();
            ends.Dispose();
            axes.Dispose();
            steps.Dispose();
            if (currentIndex == 0)
            {
                end = System.DateTime.Now;
                Console.WriteLine(end - start);
            }
        });
        // }
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
            var upPriorityK = OrtKI.Reshape(upPriority, converShape, 0).Shape[OrtKI.Reshape(upPriority, converShape, 0).Shape.Length - 1];
            var sortRet = OrtKI.TopK(OrtKI.Reshape(upPriority, converShape, 0), OrtKISharp.Tensor.MakeTensor(new long[] { upPriorityK }, new long[] { 1 }), -1, 1, 1);
            var upOrder = sortRet[1];
            var downPriorityK = OrtKI.Reshape(downPriority, converShape, 0).Shape[OrtKI.Reshape(downPriority, converShape, 0).Shape.Length - 1];
            sortRet = OrtKI.TopK(OrtKI.Reshape(downPriority, converShape, 0), OrtKISharp.Tensor.MakeTensor(new long[] { downPriorityK }, new long[] { 1 }), -1, 1, 1);
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
            var upPriorityK = OrtKI.Reshape(upPriority, converShape, 0).Shape[OrtKI.Reshape(upPriority, converShape, 0).Shape.Length - 1];
            var sortRet = OrtKI.TopK(OrtKI.Reshape(upPriority, converShape, 0), OrtKISharp.Tensor.MakeTensor(new long[] { upPriorityK }, new long[] { 1 }), -1, 1, 1);
            var upOrder = sortRet[1];
            var downPriorityK = OrtKI.Reshape(downPriority, converShape, 0).Shape[OrtKI.Reshape(downPriority, converShape, 0).Shape.Length - 1];
            sortRet = OrtKI.TopK(OrtKI.Reshape(downPriority, converShape, 0), OrtKISharp.Tensor.MakeTensor(new long[] { downPriorityK }, new long[] { 1 }), -1, 1, 1);
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
        _ = OrtKI.Reshape(roundingError, x.Shape, 0);
        _ = OrtKI.Reshape(upPriority, x.Shape, 0);
        _ = OrtKI.Reshape(downPriority, x.Shape, 0);

        return roundingNumber!;
    }
}
