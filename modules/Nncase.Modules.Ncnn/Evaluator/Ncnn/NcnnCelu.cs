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
/// Evaluator for <see cref="NcnnCelu"/>.
/// </summary>
public class NcnnCeluEvaluator : IEvaluator<NcnnCelu>, ITypeInferencer<NcnnCelu>, ICostEvaluator<NcnnCelu>, IShapeEvaluator<NcnnCelu>, IMetricEvaluator<NcnnCelu>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, NcnnCelu celu)
    {
        var input = context.GetOrtArgumentValue(celu, NcnnCelu.Input);
        var alpha = celu.Alpha;
        return OrtKI.Celu(input, alpha).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, NcnnCelu target)
    {
        var input = context.CheckArgumentType<TensorType>(target, NcnnCelu.Input);
        return Visit(input);
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, NcnnCelu target)
    {
        var ret = context.GetReturnType<TensorType>();
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(ret),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(ret),
        };
    }

    public Metric Visit(IMetricEvaluateContext context, NcnnCelu target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, NcnnCelu.Input);
        var returnType = context.GetReturnType<TensorType>();

        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(returnType) * 2,
            [MetricFactorNames.FLOPs] = MetricUtility.GetFLOPs(inputType) * (MetricUtility.DivFLOPs + MetricUtility.ExpFLOPs + MetricUtility.MulFLOPs + MetricUtility.SubFLOPs + MetricUtility.AddFLOPs + (MetricUtility.CmpFLOPs * 2)),
            [MetricFactorNames.Parallel] = 4,
        };
    }

    public Expr Visit(IShapeEvaluateContext context, NcnnCelu target) => context.GetArgumentShape(target, NcnnCelu.Input);

    private IRType Visit(TensorType input)
    {
        return input;
    }
}
