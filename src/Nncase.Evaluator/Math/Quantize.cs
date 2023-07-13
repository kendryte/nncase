// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Math;
using OrtKISharp;

namespace Nncase.Evaluator.Math;

/// <summary>
/// Evaluator for <see cref="Quantize"/>.
/// </summary>
public class QuantizeEvaluator : IEvaluator<Quantize>, ITypeInferencer<Quantize>, ICostEvaluator<Quantize>, IMetricEvaluator<Quantize>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, Quantize target)
    {
        var input = context.GetOrtArgumentValue(target, Quantize.Input);
        var quantParam = context.GetArgumentValueAsScalar<QuantParam>(target, Quantize.QuantParam);
        var zeroPoint = Tensor.FromScalar(quantParam.ZeroPoint).CastTo(target.TargetType);

        // only support qint8 in onnx
        if (input.DataType == OrtDataType.Float && target.TargetType == DataTypes.Int16)
        {
            return OrtKI.Cast((input / quantParam.Scale) + (float)quantParam.ZeroPoint, (int)OrtDataType.Int16).ToValue();
        }

        return OrtKI.QuantizeLinear(input, quantParam.Scale, zeroPoint.ToOrtTensor(), 0).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Quantize target)
    {
        var input = context.CheckArgumentType<TensorType>(target, Quantize.Input);
        var quantParam = context.CheckArgumentType<TensorType>(target, Quantize.QuantParam);
        return Visit(target, input, quantParam);
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, Quantize target)
    {
        var input = context.GetArgumentType<TensorType>(target, Quantize.Input);
        var quant_param = context.GetArgumentType<TensorType>(target, Quantize.QuantParam);
        var output = context.GetReturnType<TensorType>();

        uint macPerElement = 1;
        uint macParallel = 1;
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(input) +
                                           CostUtility.GetMemoryAccess(quant_param),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(output),
            [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(output, macPerElement) / macParallel,
        };
    }

    public Metric Visit(IMetricEvaluateContext context, Quantize target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, Quantize.Input);
        var outputType = context.GetReturnType<TensorType>();
        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(inputType) + CostUtility.GetMemoryAccess(outputType),
            [MetricFactorNames.FLOPs] = MetricUtility.GetFLOPs(outputType, 2),
        };
    }

    private IRType Visit(Quantize target, TensorType input, TensorType quantParam)
    {
        return new TensorType(target.TargetType, input.Shape);
    }
}
