// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.NN;
using OrtKISharp;

namespace Nncase.Evaluator.NN;

/// <summary>
/// Evaluator for <see cref="LayerNorm"/>.
/// </summary>
public class LayerNormEvaluator : IEvaluator<LayerNorm>, ITypeInferencer<LayerNorm>, ICostEvaluator<LayerNorm>, IShapeEvaluator<LayerNorm>, IMetricEvaluator<LayerNorm>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, LayerNorm layerNorm)
    {
        var input = context.GetOrtArgumentValue(layerNorm, LayerNorm.Input);
        var scale = context.GetOrtArgumentValue(layerNorm, LayerNorm.Scale);
        var bias = context.GetOrtArgumentValue(layerNorm, LayerNorm.Bias);

        // return Value.FromTensor(OrtKI.LayerNormalization(input, scale, bias, layerNorm.Axis, layerNorm.Epsilon, 1));
        return Value.FromTensor(LayerNormImpl(input.ToTensor(), scale.ToTensor(), bias.ToTensor(), layerNorm.Axis, layerNorm.Epsilon));
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, LayerNorm target)
    {
        var input = context.CheckArgumentType<TensorType>(target, LayerNorm.Input);
        return Visit(input);
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, LayerNorm target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, LayerNorm.Input);
        var returnType = context.GetReturnType<TensorType>();
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(inputType),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(returnType),
        };
    }

    public Metric Visit(IMetricEvaluateContext context, LayerNorm target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, LayerNorm.Input);
        var returnType = context.GetReturnType<TensorType>();

        var r = MetricUtility.GetFLOPs(returnType);
        var i = MetricUtility.GetFLOPs(inputType);
        var outter = i / r;
        var inner = i / outter;

        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(inputType) + CostUtility.GetMemoryAccess(returnType),
            [MetricFactorNames.FLOPs] = outter * ((inner * 7) + MetricUtility.SqrtFLOPs),
            [MetricFactorNames.Parallel] = 4,
        };
    }

    public Expr Visit(IShapeEvaluateContext context, LayerNorm target) => context.GetArgumentShape(target, LayerNorm.Input);

    private IRType Visit(TensorType input)
    {
        return input;
    }

#if true
    private Tensor LayerNormImpl(Tensor input, Tensor scale, Tensor bias, int axis, float epsilon)
    {
        int outputSize = 1;
        int innerSize = 1;
        float[] inputArray = input.ToArray<float>();
        float[] outputArray = new float[inputArray.Length];
        int[] inShape = input.Shape.ToValueArray();
        for (int i = 0; i < axis; i++)
        {
            outputSize *= inShape[i];
        }

        for (int i = axis; i < inShape.Length; i++)
        {
            if (i < 0)
            {
                innerSize *= inShape[^System.Math.Abs(i)];
            }
            else
            {
                innerSize *= inShape[i];
            }
        }

        for (int batch = 0; batch < outputSize; batch++)
        {
            float mean1 = 0f;
            for (int i = 0; i < innerSize; i++)
            {
                mean1 += inputArray[(i + (batch * innerSize)) % inputArray.Length] / innerSize;
            }

            float[] sub = new float[innerSize];
            for (int i = 0; i < innerSize; i++)
            {
                sub[i] = inputArray[(i + (batch * innerSize)) % inputArray.Length] - mean1;
            }

            float[] pow = new float[innerSize];
            for (int i = 0; i < innerSize; i++)
            {
                pow[i] = (float)System.Math.Pow(sub[i], 2);
            }

            float mean2 = 0f;
            for (int i = 0; i < innerSize; i++)
            {
                mean2 += pow[i] / innerSize;
            }

            float add = mean2 + epsilon;
            float sqrt = (float)System.Math.Sqrt(add);

            float[] div = new float[innerSize];
            for (int i = 0; i < innerSize; i++)
            {
                div[i] = sub[i] / sqrt;
            }

            for (int i = 0; i < innerSize; i++)
            {
                outputArray[(i + (batch * innerSize)) % outputArray.Length] = (div[i] * scale.ToArray<float>()[i % scale.Length]) + bias.ToArray<float>()[i % bias.Length];
            }
        }

        return new Tensor<float>(outputArray, input.Shape);
    }
#endif
}
