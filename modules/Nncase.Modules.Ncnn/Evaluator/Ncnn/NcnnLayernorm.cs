// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Ncnn;
using OrtKISharp;

namespace Nncase.Evaluator.Ncnn;

/// <summary>
/// Evaluator for <see cref="NcnnLayerNorm"/>.
/// </summary>
public class NcnnLayerNormEvaluator : IEvaluator<NcnnLayerNorm>, ITypeInferencer<NcnnLayerNorm>, ICostEvaluator<NcnnLayerNorm>, IShapeEvaluator<NcnnLayerNorm>, IMetricEvaluator<NcnnLayerNorm>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, NcnnLayerNorm layerNorm)
    {
        var input = context.GetOrtArgumentValue(layerNorm, NcnnLayerNorm.Input);

        return OrtKI.LayerNormalization(input, layerNorm.GammaData, layerNorm.BetaData, layerNorm.AffineSize, layerNorm.Eps, 1L)[0].ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, NcnnLayerNorm target)
    {
        var input = context.CheckArgumentType<TensorType>(target, NcnnLayerNorm.Input);
        return Visit(input);
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, NcnnLayerNorm target)
    {
        var ret = context.GetReturnType<TensorType>();
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(ret),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(ret),
        };
    }

    public Metric Visit(IMetricEvaluateContext context, NcnnLayerNorm target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, NcnnLayerNorm.Input);
        var gamma = new TensorType(DataTypes.Float32, new[] { target.AffineSize });
        var returnType = context.GetReturnType<TensorType>();

        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = (CostUtility.GetMemoryAccess(returnType) * 2) + (CostUtility.GetMemoryAccess(gamma) * 2),

            // x = (x - mean)/(standard_deviation) * gamma + beta;
            // mean = sum(x)/N;
            // standard-deviation = sqrt(sum(square(x-mean))/N + eps);
            [MetricFactorNames.FLOPs] =
                (MetricUtility.GetFLOPs(inputType) * (MetricUtility.AddFLOPs + MetricUtility.SubFLOPs)) + MetricUtility.DivFLOPs + // x = x-mean
                (MetricUtility.GetFLOPs(inputType) * (MetricUtility.PowFLOPs + MetricUtility.AddFLOPs)) + MetricUtility.SqrtFLOPs + MetricUtility.DivFLOPs + MetricUtility.AddFLOPs +
                (MetricUtility.GetFLOPs(inputType) * (MetricUtility.DivFLOPs + MetricUtility.MulFLOPs + MetricUtility.AddFLOPs)),
            [MetricFactorNames.Parallel] = 4,
        };
    }

    public Expr Visit(IShapeEvaluateContext context, NcnnLayerNorm target) => context.GetArgumentShape(target, NcnnLayerNorm.Input);

    private IRType Visit(TensorType input)
    {
        return input;
    }
}
