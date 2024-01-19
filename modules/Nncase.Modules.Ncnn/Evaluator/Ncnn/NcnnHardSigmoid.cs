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
/// Evaluator for <see cref="NcnnHardSigmoid"/>.
/// </summary>
public class NcnnHardSigmoidEvaluator : IEvaluator<NcnnHardSigmoid>, ITypeInferencer<NcnnHardSigmoid>, ICostEvaluator<NcnnHardSigmoid>, IShapeEvaluator<NcnnHardSigmoid>, IMetricEvaluator<NcnnHardSigmoid>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, NcnnHardSigmoid hardSigmoid)
    {
        var input = context.GetOrtArgumentValue(hardSigmoid, NcnnHardSigmoid.Input);
        var alpha = hardSigmoid.Alpha;
        var beta = hardSigmoid.Beta;
        return OrtKI.HardSigmoid(input, alpha, beta).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, NcnnHardSigmoid target)
    {
        var input = context.CheckArgumentType<TensorType>(target, NcnnHardSigmoid.Input);
        return Visit(input);
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, NcnnHardSigmoid target)
    {
        var ret = context.GetReturnType<TensorType>();
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(ret),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(ret),
        };
    }

    public Metric Visit(IMetricEvaluateContext context, NcnnHardSigmoid target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, NcnnHardSigmoid.Input);
        var returnType = context.GetReturnType<TensorType>();

        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(returnType) * 2,
            [MetricFactorNames.FLOPs] = MetricUtility.GetFLOPs(inputType) * (MetricUtility.ExpFLOPs + MetricUtility.MulFLOPs + MetricUtility.SubFLOPs + MetricUtility.CmpFLOPs),
            [MetricFactorNames.Parallel] = 4,
        };
    }

    public Expr Visit(IShapeEvaluateContext context, NcnnHardSigmoid target) => context.GetArgumentShape(target, NcnnHardSigmoid.Input);

    private IRType Visit(TensorType input)
    {
        return input;
    }
}
