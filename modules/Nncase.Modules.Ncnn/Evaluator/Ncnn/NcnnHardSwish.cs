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
/// Evaluator for <see cref="NcnnHardSwish"/>.
/// </summary>
public class NcnnHardSwishEvaluator : IEvaluator<NcnnHardSwish>, ITypeInferencer<NcnnHardSwish>, ICostEvaluator<NcnnHardSwish>, IShapeEvaluator<NcnnHardSwish>, IMetricEvaluator<NcnnHardSwish>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, NcnnHardSwish hardSwish)
    {
        var input = context.GetOrtArgumentValue(hardSwish, NcnnHardSwish.Input);
        return OrtKI.HardSwish(input).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, NcnnHardSwish target)
    {
        var input = context.CheckArgumentType<TensorType>(target, NcnnHardSwish.Input);
        return Visit(input);
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, NcnnHardSwish target)
    {
        var ret = context.GetReturnType<TensorType>();
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(ret),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(ret),
        };
    }

    public Metric Visit(IMetricEvaluateContext context, NcnnHardSwish target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, NcnnHardSwish.Input);
        var returnType = context.GetReturnType<TensorType>();

        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(returnType) * 2,
            [MetricFactorNames.FLOPs] = MetricUtility.GetFLOPs(inputType) * (MetricUtility.PowFLOPs + MetricUtility.MulFLOPs + MetricUtility.AddFLOPs + (MetricUtility.CmpFLOPs * 2)),
            [MetricFactorNames.Parallel] = 4,
        };
    }

    public Expr Visit(IShapeEvaluateContext context, NcnnHardSwish target) => context.GetArgumentShape(target, NcnnHardSwish.Input);

    private IRType Visit(TensorType input)
    {
        return input;
    }
}
