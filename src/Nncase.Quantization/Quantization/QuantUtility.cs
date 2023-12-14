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
using Nncase.TIR.Builders;
using OrtKISharp;
using SMath = System.Math;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.F.Math;

namespace Nncase.Quantization;

/// <summary>
/// QuantAlgorithm utility.
/// </summary>
public static class QuantAlgorithmUtility
{
    public static Tensor<float> SquantWeights(Tensor<float> inputWeights, Tensor<float> inputWeightsRanges,
        ReadOnlySpan<int> inputWeightsShape, QuantMode quantMode, int bits, bool isByChannel)
    {
        (float qMax, float qMin) = GetQRange(quantMode, bits);

        if (!isByChannel)
        {
            throw new NotSupportedException("By tensor weights quant is not supported.");
        }
        var x = inputWeights;
        var shape = inputWeightsShape.ToArray();
        var outChannel = inputWeightsShape[0];

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

        var delta = Tensor.From(deltaArr, shape);
        var zeroPoint = Tensor.From(zeroPointArr, shape);

        var quantTensor = (((Expr)x / delta) + zeroPoint).Evaluate().AsTensor();
        var xInt = AdaptiveRound(quantTensor, qMin, qMax); // SQuant量化
        var xQuant = OrtKI.Clip(xInt.ToOrtTensor(), OrtKISharp.Tensor.FromScalar(qMin),
            OrtKISharp.Tensor.FromScalar(qMax)).ToTensor();
        var xDequant = (((Expr)xQuant - zeroPoint) * delta).Evaluate().AsTensor();

        return Tensor.From<float>(xDequant.ToArray<float>(), inputWeights.Shape);
    }

    private static (float qMax, float qMin) GetQRange(QuantMode quantMode, int bits)
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

        return (qMax, qMin);
    }

    private static (Tensor Number, Tensor Error, Tensor Priority, Tensor Priority1) RoundingForward(float roundingErrorSum, Tensor roundingNumber, Tensor roundingError, SQuantParam param, Tensor priority, Tensor priority1)
    {
        var roundingNumberMem = roundingNumber.ToArray<float>();
        var roundingErrorMem = roundingError.ToArray<float>();
        var priorityMem = priority.ToArray<float>();
        var priority1Mem = priority1.ToArray<float>();
        int topK = (int)System.Math.Round(System.Math.Abs(roundingErrorSum));
        bool overSquant = topK >= System.Math.Abs(roundingErrorSum);
        if (topK > 0)
        {
            var starts = Tensor.From(new long[] { 0 });
            var ends = Tensor.From(new long[] { topK });
            var axes = Tensor.From(new long[] { 0 });
            var steps = Tensor.From(new long[] { 1 });

            var orderTmp = Slice(param.Order, starts, ends, axes, steps).Evaluate().AsTensor();

            var orderTmpArr = orderTmp.ToArray<long>();
            var orderArr = param.Order.ToArray<int>();
            var errorArr = param.Error.ToArray<float>();
            var numberArr = param.Number.ToArray<float>();
            for (int i = 0; i < orderTmp.Length; i++)
            {
                var index = orderTmpArr[i];
                roundingErrorMem[(int)index] = errorArr[index];
                roundingNumberMem[(int)index] = numberArr[index];
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
        }
        // roundingNumberTmp, roundingErrorTmp, priorityTmp, priority1Tmp
        return (
            Tensor.From(roundingNumberMem, roundingNumber.Shape),
            Tensor.From(roundingErrorMem, roundingError.Shape),
            Tensor.From(priorityMem, priority.Shape),
            Tensor.From(priority1Mem, priority1.Shape)
            );
    }

    private static (Tensor Number, Tensor Error, Tensor Priority, Tensor Priority1)  SQuantFunc(Tensor roundingErrorSum, Tensor roundingNumberOrigin,
        Tensor roundingErrorOrigin, SQuantParam upParam, SQuantParam downParam, long[] converShape)
    {
        var upNumber = Reshape(upParam.Number, converShape).Evaluate().AsTensor();
        var upError = Reshape(upParam.Error, converShape).Evaluate().AsTensor();
        var upPriority = Reshape(upParam.Priority!, converShape).Evaluate().AsTensor();
        var downNumber = Reshape(downParam.Number, converShape).Evaluate().AsTensor();
        var downError = Reshape(downParam.Error, converShape).Evaluate().AsTensor();
        var downPriority = Reshape(downParam.Priority!, converShape).Evaluate().AsTensor();
        var upOrder = upParam.Order;
        var downOrder = downParam.Order;
        var roundingNumber = Reshape(roundingNumberOrigin, converShape).Evaluate().AsTensor();
        var roundingError = Reshape(roundingErrorOrigin, converShape).Evaluate().AsTensor();

        var roundingNumberShape = roundingNumber.Shape.ToValueArray();
        var batches = roundingNumberShape[0];
        var inputChannel = roundingNumberShape[1];

        var totalSize = roundingNumberShape.Aggregate((sum, x) => sum * x);
        var oneBatchSize = totalSize / batches;
        // var oneInputChannelSize = oneBatchSize / inputChannel;

        // todo: update this tensor, return
        var roundingErrorSumArr = roundingErrorSum.ToArray<float>();

        // Console.WriteLine(batches);
        // Console.WriteLine(inputChannel);
        // var data = Enumerable.Range(0, (int)batches).SelectMany(n =>
        // {
        var index = Enumerable.Range(0, batches * inputChannel).AsParallel();
        var data = ParallelEnumerable.Select<int, (Tensor RoundingNumber, Tensor RoundingError, Tensor Priority, Tensor Priority1)>(index, currentIndex =>
            // Parallel.For(0, inputChannel, c =>
        {
            var n = currentIndex / inputChannel;
            var c = currentIndex % inputChannel;
            var starts = Tensor.From(new long[] { n, c }, new[] { 2 });
            var ends = Tensor.From(new long[] { n + 1, c + 1 }, new[] { 2 });
            var axes = Tensor.From(new long[] { 0, 1 }, new[] { 2 });
            var steps = Tensor.From(new long[] { 1, 1 }, new[] { 2 });
            Func<Nncase.Tensor, Nncase.Tensor> doSlice = tensor =>
            {
                return Squeeze(Slice(tensor, starts, ends, axes, steps), axes).Evaluate().AsTensor();
            };
            var roundingNumberTmp = doSlice(roundingNumber);
            var roundingErrorTmp = doSlice(roundingError);
            var upNumberSlice = doSlice(upNumber);
            var upErrorSlice = doSlice(upError);
            var upOrderSlice = doSlice(upOrder);
            var downNumberSlice = doSlice(downNumber);
            var downErrorSlice = doSlice(downError);
            var downOrderSlice = doSlice(downOrder);

            var isDown = roundingErrorSumArr[(n * inputChannel) + c] < 0;
            var roundingErrorSumValue = roundingErrorSumArr[(n * inputChannel) + c];
            var priorityTmp = doSlice(upPriority);
            var priority1Tmp = doSlice(downPriority);
            if (isDown)
            {
                var upParamSlice = new SQuantParam(upNumberSlice, upErrorSlice, null, upOrderSlice);
                return RoundingForward(roundingErrorSumValue, roundingNumberTmp, roundingErrorTmp,
                    upParamSlice, priorityTmp, priority1Tmp);
            }
            else
            {
                // todo: priority order??
                var downParamSlice = new SQuantParam(downNumberSlice, downErrorSlice, null, downOrderSlice);
                return RoundingForward(roundingErrorSumValue, roundingNumberTmp, roundingErrorTmp,
                    downParamSlice, priority1Tmp, priorityTmp);
            }
        }).ToArray();
        var number = Reshape(Concat(new IR.Tuple(data.Select(x => (Expr)x.RoundingNumber).ToArray()), 0), roundingNumber.Shape).Evaluate().AsTensor();
        var error = Reshape(Concat(new IR.Tuple(data.Select(x => (Expr)x.RoundingError).ToArray()), 0), roundingError.Shape).Evaluate().AsTensor();
        var priority = Reshape(Concat(new IR.Tuple(data.Select(x => (Expr)x.Priority).ToArray()), 0), upPriority.Shape).Evaluate().AsTensor();
        var priority1 = Reshape(Concat(new IR.Tuple(data.Select(x => (Expr)x.Priority1).ToArray()), 0), downPriority.Shape).Evaluate().AsTensor();
        return (number, error, priority, priority1);
    }

    record SQuantParam(Tensor Number, Tensor Error, Tensor? Priority, Tensor Order);

    private static Tensor AdaptiveRound(Tensor tempX, float tMin, float tMax)
    {
        var x = tempX;
        var roundingNumber = Round(x).Evaluate().AsTensor(); // round取整值
        var roundingError = ((Expr)roundingNumber - x).Evaluate().AsTensor(); // 误差

        var xTmp = Reshape(x, new long[] { x.Shape[0].FixedValue, x.Shape[1].FixedValue, -1 }).Evaluate().AsTensor();
        var tmpConverShape = xTmp.Shape; // HW维度合并
        var squantK = tmpConverShape[2] != 1; // 只有一个元素时， 不做K的逼近

        var converShape = squantK ? xTmp.Shape.Select(x => (long)x.FixedValue).ToArray() : new long[] { 1, x.Shape[0].FixedValue, -1 };
        var roundingErrorSum =
            ReduceSum(Reshape(roundingError, converShape), new long[] { -1 }, 0, 0);
        var upParam = UpData(tMax, roundingNumber, roundingError, x, squantK, converShape);
        var downParam = DownData(tMin, roundingNumber, roundingError, x, squantK, converShape);
        var (num, _, _, _) = SQuantFunc(roundingErrorSum.Evaluate().AsTensor(), roundingNumber, roundingError, upParam, downParam, converShape);
        return num;
    }

    private static Tensor GetDownOrder(Tensor Priority, long[] converShape)
    {
        var reshapePriority = Reshape(Priority, converShape).Evaluate().AsTensor();
        var downPriorityK = reshapePriority.Shape[^1].FixedValue;
        var tmpsortRet = TopK(reshapePriority,
            Tensor.From(new long[] { downPriorityK }), -1, 1, 1);
        var downOrder = tmpsortRet[1].Evaluate().AsTensor();
        return downOrder;
    }

    private static Tensor GetUpOrder(Tensor Priority, long[] converShape)
    {
        var reshapePriority = Reshape(Priority, converShape).Evaluate().AsTensor();
        var upPriorityK = reshapePriority.Shape[^1].FixedValue;
        var sortRet = TopK(reshapePriority, Tensor.From(new long[] { upPriorityK }), -1, 1, 1);
        var upOrder = sortRet[1].Evaluate().AsTensor();
        return upOrder;
    }

    private static SQuantParam DownData(float tMin, Tensor roundingNumberOrigin,
        Tensor roundingError, Tensor originX, bool squantK, long[] converShape)
    {
        var x = originX;
        Expr downNumberTensor = roundingNumberOrigin;
        Expr downError = roundingError;
        downError = Where(LessThan(x, tMin), 0f, downError); // 边界上的值不能再调整，所以去除
        downError = Where(LessThan(downError, 0f), 0f, downError); // 误差为负的都设为0，即down对应“原值<量化值”的集合
        var downPriority = Abs(downError);

        downError = Where(LogicalNot(Equal(downError, 0.0f)), downError - 1.0f,
            downError); // down集合中，Flip翻转后对应的误差
        var downNumber =
            Where(LogicalNot(Equal(downError, 0.0f)), downNumberTensor - 1.0f, downNumberTensor).Evaluate().AsTensor().ToOrtTensor(); // down集合中，Flip翻转后对应的取整值
        if (squantK)
        {
            downPriority *= 0.0f;
        }

        var downPriorityResult = downPriority.Evaluate().AsTensor();
        var downOrder = GetDownOrder(downPriorityResult, converShape);
        return new(downNumber.ToTensor(), downError.Evaluate().AsTensor(), downPriorityResult, downOrder);
    }

    private static SQuantParam UpData(float tMax, Tensor roundingNumber, Tensor roundingError, Tensor x, bool squantK, long[] converShape)
    {
        Expr upNumber = roundingNumber;
        Expr upError = roundingError;
        upError = Where(GreaterThan(x, tMax), 0f, upError); // 边界上的值不能再调整，所以去除
        upError = Where(GreaterThan(upError, 0.0f), 0f, upError); // 误差为正的都设为0，即up对应“原值>量化值”的集合
        var upPriority = Abs(upError);

        upError = Where(LogicalNot(Equal(upError, 0.0f)), upError + 1.0f, upError); // up集合中，Flip翻转后对应的误差
        upNumber = Where(LogicalNot(Equal(upError, 0.0f)), upNumber + 1.0f, upNumber); // up集合中，Flip翻转后对应的取整值
        if (squantK)
        {
            upPriority *= 0.0f;
        }

        var upPriorityValue = upPriority.Evaluate().AsTensor();
        var upOrder = GetUpOrder(upPriorityValue, converShape);
        return new(upNumber.Evaluate().AsTensor(), upError.Evaluate().AsTensor(), upPriorityValue, upOrder);
    }
}
