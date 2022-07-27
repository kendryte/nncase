using System;
using System.Collections.Generic;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.K210;
using Nncase.Evaluator.K210;
using Nncase.IR.Math;
using static Nncase.Evaluator.EvaluatorUtil;

namespace Nncase.Evaluator.K210;

/// <summary>
/// Evaluator for <see cref="KPUConv2D"/>.
/// </summary>
public class KPUConv2DEvaluator : IEvaluator<KPUConv2D>, ITypeInferencer<KPUConv2D>, ICostEvaluator<KPUConv2D>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, KPUConv2D conv)
    {
        var input = context.GetTFArgumentValue(conv, KPUConv2D.Input);
        var weights = context.GetTFArgumentValue(conv, KPUConv2D.Weights);
        var ortArgumentValue = context.GetTFArgumentValue(conv, KPUConv2D.BatchNorms);
        var argumentValue = context.GetTFArgumentValue(conv, KPUConv2D.OutputQuantParam);
        var inShape = conv.CheckedShape;
        var outShape = conv.CheckedShape;
        var inChannels = inShape[1].FixedValue;
        var outChannels = outShape[1].FixedValue;
        var argX = context.GetTFArgumentValue(conv, KPUConv2D.ArgX);
        var argW = context.GetTFArgumentValue(conv, KPUConv2D.ArgW);
        var shiftX = context.GetTFArgumentValue(conv, KPUConv2D.ShiftX);
        var shiftW = context.GetTFArgumentValue(conv, KPUConv2D.ShiftW);
        var argAdd = context.GetTFArgumentValue(conv, KPUConv2D.ArgAdd);
        var groups = 1L;
        var pad = KPUUtility.GetKPUPadding(conv.FilterType);
        var fliter = KPUUtility.GetKPUFilter(conv.FilterType);
        var isDepthwise = inChannels == outChannels && outChannels == groups;
        return kernel.KPUConv2D(input, weights, inShape[2].FixedValue, inShape[3].FixedValue, inChannels,
            outChannels, pad, 0, 0, 0, 0, 0, fliter,
            isDepthwise, KPUUtility.Activation(), KPUUtility.BatchNorm());
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, KPUConv2D target)
    {
        var input = context.CheckArgumentType<TensorType>(target, KPUConv2D.Input);
        var weights = context.CheckArgumentType<TensorType>(target, KPUConv2D.Weights);
        var ortArgumentValue = context.CheckArgumentType<TensorType>(target, KPUConv2D.BatchNorms);
        var argumentValue = context.CheckArgumentType<TensorType>(target, KPUConv2D.OutputQuantParam);
        return Visit(context, target, input, weights, ortArgumentValue, argumentValue);
    }


    /// <inheritdoc/>
    public Cost? Visit(ICostEvaluateContext context, KPUConv2D target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, KPUConv2D.Input);
        var weightsType = context.GetArgumentType<TensorType>(target, KPUConv2D.Weights);
        var weightsShape = context.GetArgumentType<TensorType>(target, KPUConv2D.Weights).Shape;
        var outputType = context.GetReturnType<TensorType>();

        if (weightsShape.IsFixed)
        {
            var macPerElement = weightsShape[1] * weightsShape[2] * weightsShape[3];
            var kpuMac = target.FilterType == KPUFilterType.Filter_1x1 ? 64 : 64 * 9;
            return new()
             {
                 [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(inputType) + CostUtility.GetMemoryAccess(weightsType),
                 [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(outputType),
                 [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(outputType, macPerElement.FixedValue * 2) / kpuMac,
             };
        }

        return null;
    }

    private IRType Visit(ITypeInferenceContext context, KPUConv2D target, TensorType input, TensorType weights, TensorType ortArgumentValue, TensorType argumentValue)
    {
        var stride = new[] { 1, 1 };
        var pad = Tensor.FromScalar(KPUUtility.GetKPUPadding(target.FilterType), new[] { 2, 2 });
        var dilation = new[] { 1, 1 };
        var groups = 1;
        return TypeInference.Conv2DType(input, weights, stride, pad, dilation, groups);
    }
}