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
/// Evaluator for <see cref="NcnnBatchNorm"/>.
/// </summary>
public class NcnnBatchNormEvaluator : IEvaluator<NcnnBatchNorm>, ITypeInferencer<NcnnBatchNorm>, ICostEvaluator<NcnnBatchNorm>, IShapeEvaluator<NcnnBatchNorm>, IMetricEvaluator<NcnnBatchNorm>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, NcnnBatchNorm batchnorm)
    {
        var input = context.GetOrtArgumentValue(batchnorm, NcnnBatchNorm.Input);
        var eps = batchnorm.Eps;
        var slopeData = batchnorm.SlopeData.ToArray();
        var meanData = batchnorm.MeanData.ToArray();
        var varData = batchnorm.VarData.ToArray();
        var biasData = batchnorm.BiasData.ToArray();
        return OrtKI.BatchNormalization(input, slopeData, biasData, meanData, varData, eps, 0.0f).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, NcnnBatchNorm target)
    {
        var input = context.CheckArgumentType<TensorType>(target, NcnnBatchNorm.Input);
        return Visit(input);
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, NcnnBatchNorm target)
    {
        var ret = context.GetReturnType<TensorType>();
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(ret),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(ret),
        };
    }

    public Metric Visit(IMetricEvaluateContext context, NcnnBatchNorm target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, NcnnBatchNorm.Input);
        var returnType = context.GetReturnType<TensorType>();
        var returnF = MetricUtility.GetFLOPs(returnType);
        var inputF = MetricUtility.GetFLOPs(inputType);
        var inner = inputF / returnF;

        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(returnType) * 2,
            [MetricFactorNames.FLOPs] = (inner * 2) + (inputF * (MetricUtility.SubFLOPs + MetricUtility.DivFLOPs + MetricUtility.MulFLOPs + MetricUtility.AddFLOPs)),
            [MetricFactorNames.Parallel] = 4,
        };
    }

    public Expr Visit(IShapeEvaluateContext context, NcnnBatchNorm target) => context.GetArgumentShape(target, NcnnBatchNorm.Input);

    private IRType Visit(TensorType input)
    {
        return input;
    }
}
