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
/// Evaluator for <see cref="NcnnInstanceNorm"/>.
/// </summary>
public class NcnnInstanceNormEvaluator : IEvaluator<NcnnInstanceNorm>, ITypeInferencer<NcnnInstanceNorm>, ICostEvaluator<NcnnInstanceNorm>, IShapeEvaluator<NcnnInstanceNorm>, IMetricEvaluator<NcnnInstanceNorm>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, NcnnInstanceNorm instanceNorm)
    {
        var input = context.GetOrtArgumentValue(instanceNorm, NcnnInstanceNorm.Input);

        return OrtKI.InstanceNormalization(input, instanceNorm.GammaData, instanceNorm.BetaData, instanceNorm.Eps).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, NcnnInstanceNorm target)
    {
        var input = context.CheckArgumentType<TensorType>(target, NcnnInstanceNorm.Input);
        return Visit(input);
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, NcnnInstanceNorm target)
    {
        var ret = context.GetReturnType<TensorType>();
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(ret),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(ret),
        };
    }

    public Metric Visit(IMetricEvaluateContext context, NcnnInstanceNorm target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, NcnnInstanceNorm.Input);
        var gamma = new TensorType(DataTypes.Float32, new[] { target.Channels });
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

    public Expr Visit(IShapeEvaluateContext context, NcnnInstanceNorm target) => context.GetArgumentShape(target, NcnnInstanceNorm.Input);

    private IRType Visit(TensorType input)
    {
        return input;
    }
}
