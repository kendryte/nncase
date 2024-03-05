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
/// Evaluator for <see cref="NcnnDequantize"/>.
/// </summary>
public class NcnnDequantizeEvaluator : IEvaluator<NcnnDequantize>, ITypeInferencer<NcnnDequantize>, ICostEvaluator<NcnnDequantize>, IShapeEvaluator<NcnnDequantize>, IMetricEvaluator<NcnnDequantize>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, NcnnDequantize dequantize)
    {
        var input = context.GetOrtArgumentValue(dequantize, NcnnDequantize.Input);
        return OrtKI.DequantizeLinear(input, dequantize.Scale, dequantize.Bias, 0).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, NcnnDequantize target)
    {
        var input = context.CheckArgumentType<TensorType>(target, NcnnDequantize.Input);
        return Visit(input);
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, NcnnDequantize target)
    {
        var input = context.GetArgumentType<TensorType>(target, NcnnDequantize.Input);
        var ret = context.GetReturnType<TensorType>();
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(input) +
                                           (UInt128)((target.Scale.Length + target.Bias.Length) * sizeof(float)),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(ret),
        };
    }

    public Metric Visit(IMetricEvaluateContext context, NcnnDequantize target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, NcnnDequantize.Input);
        var outputType = context.GetReturnType<TensorType>();

        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(inputType) + CostUtility.GetMemoryAccess(outputType),
            [MetricFactorNames.FLOPs] = MetricUtility.GetFLOPs(outputType, 2),
        };
    }

    public Expr Visit(IShapeEvaluateContext context, NcnnDequantize target) => context.GetArgumentShape(target, NcnnDequantize.Input);

    private IRType Visit(TensorType input)
    {
        // ncnn only support dequantize to float32.
        return new TensorType(DataTypes.Float32, input.Shape);
    }
}
