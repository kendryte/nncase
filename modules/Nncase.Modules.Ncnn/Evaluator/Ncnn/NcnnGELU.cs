// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NetFabric.Hyperlinq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.Ncnn;
using Nncase.IR.NN;
using OrtKISharp;

namespace Nncase.Evaluator.Ncnn;

/// <summary>
/// Evaluator for <see cref="NcnnGELU"/>.
/// </summary>
public class NcnnGELUEvaluator : IEvaluator<NcnnGELU>, ITypeInferencer<NcnnGELU>, ICostEvaluator<NcnnGELU>, IShapeEvaluator<NcnnGELU>, IMetricEvaluator<NcnnGELU>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, NcnnGELU celu)
    {
        var input = context.GetOrtArgumentValue(celu, NcnnGELU.Input);
        var res = IR.F.Math.Mul(IR.F.NN.Erf(IR.F.Math.Div(input.ToValue().AsTensor(), 1.4142135381698608)) + 1 , input.ToValue().AsTensor()) * 0.5;
        return res.Evaluate();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, NcnnGELU target)
    {
        var input = context.CheckArgumentType<TensorType>(target, NcnnGELU.Input);
        return Visit(input);
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, NcnnGELU target)
    {
        var ret = context.GetReturnType<TensorType>();
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(ret),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(ret),
        };
    }

    public Metric Visit(IMetricEvaluateContext context, NcnnGELU target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, NcnnGELU.Input);
        var returnType = context.GetReturnType<TensorType>();

        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(returnType) * 2,
            [MetricFactorNames.Parallel] = 4,
        };
    }

    public Expr Visit(IShapeEvaluateContext context, NcnnGELU target) => context.GetArgumentShape(target, NcnnGELU.Input);

    private IRType Visit(TensorType input)
    {
        return input;
    }
}
