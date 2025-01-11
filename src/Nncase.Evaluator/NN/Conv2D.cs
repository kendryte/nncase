// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.NN;
using OrtKISharp;
using static Nncase.Evaluator.EvaluatorUtil;
using static Nncase.IR.F.Tensors;

namespace Nncase.Evaluator.NN;

/// <summary>
/// Evaluator for <see cref="Conv2D"/>.
/// </summary>
public class Conv2DEvaluator : IEvaluator<Conv2D>, ITypeInferencer<Conv2D>, ICostEvaluator<Conv2D>, IShapeEvaluator<Conv2D>, IMetricEvaluator<Conv2D>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, Conv2D conv)
    {
        var input = context.GetOrtArgumentValue(conv, Conv2D.Input);
        var weights = context.GetOrtArgumentValue(conv, Conv2D.Weights);
        var bias = context.GetOrtArgumentValue(conv, Conv2D.Bias);

        var stride = context.GetArgumentValueAsArray<long>(conv, Conv2D.Stride);
        var pad = context.GetInt64OrtTensorArgumentValue(conv, Conv2D.Padding);
        var dilation = context.GetArgumentValueAsArray<long>(conv, Conv2D.Dilation);
        var groups = context.GetArgumentValueAsScalar<long>(conv, Conv2D.Groups);
        var fusedClamp = context.GetArgumentValueAsArray<float>(conv, Conv2D.FusedClamp);
        var kernelShape = weights.Shape;
        var result = OrtKI.Conv(
            input,
            weights,
            bias,
            "NOTSET",
            dilation,
            groups,
            new long[] { kernelShape[2], kernelShape[3] },
            ToOnnxPadFormat(pad),
            stride);
        var outType = input.ToTensor().ElementType;
        return Value.FromTensor(OrtKI.Clip(result.ToTensor().Cast<float>().ToOrtTensor(), fusedClamp[0], fusedClamp[1]).ToTensor().CastTo(outType));
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Conv2D target)
    {
        var input = context.GetArgumentType(target, Conv2D.Input);
        var weights = context.GetArgumentType(target, Conv2D.Weights);
        var bias = context.GetArgumentType(target, Conv2D.Bias);
        return (input, weights) switch
        {
            (DistributedType a, DistributedType b) => Visit(context, target, a, b, (DistributedType)bias),
            (TensorType a, TensorType b) => Visit(context, target, a, b),
            (AnyType, _) => AnyType.Default,
            (_, AnyType) => AnyType.Default,
            _ => new InvalidType(string.Empty),
        };
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, Conv2D target)
    {
        var inputType = context.GetArgumentType<IRType>(target, Conv2D.Input);
        var weightsType = context.GetArgumentType<IRType>(target, Conv2D.Weights);
        var biasType = context.GetArgumentType<IRType>(target, Conv2D.Bias);
        var outputType = context.GetReturnType<IRType>();

        var weightsShape = weightsType is TensorType ? ((TensorType)weightsType).Shape : ((DistributedType)weightsType).TensorType.Shape;
        var macPerElement = (2 * weightsShape[1] * weightsShape[2] * weightsShape[3]) - 1;
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(inputType) + CostUtility.GetMemoryAccess(weightsType) + CostUtility.GetMemoryAccess(biasType),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(outputType),
            [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(outputType, (uint)macPerElement.FixedValue),
        };
    }

    public Metric Visit(IMetricEvaluateContext context, Conv2D target)
    {
        var returnType = context.GetReturnType<TensorType>();
        var outputShape = returnType.Shape.ToValueArray();

        var inputType = context.GetArgumentType<TensorType>(target, Conv2D.Input);
        var inputShape = inputType.Shape.ToValueArray();
        var weightType = context.GetArgumentType<TensorType>(target, Conv2D.Weights);
        var weightShape = weightType.Shape.ToValueArray();

        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(inputType) + CostUtility.GetMemoryAccess(weightType) + CostUtility.GetMemoryAccess(returnType),
            [MetricFactorNames.FLOPs] = (UInt128)(inputShape[0] * weightShape[0] * weightShape[1] * outputShape[2] * outputShape[3] * weightShape[2] * weightShape[3]),
        };
    }

    public Expr Visit(IShapeEvaluateContext context, Conv2D target)
    {
        var input = context.GetArgumentShape(target, Conv2D.Input);
        var weights = context.GetArgumentShape(target, Conv2D.Weights);
        var pad = context.GetArgument(target, Conv2D.Padding);
        var stride = context.GetArgument(target, Conv2D.Stride);
        var dilation = context.GetArgument(target, Conv2D.Dilation);
        var groups = context.GetArgument(target, Conv2D.Groups);
        return IR.F.ShapeExpr.Conv2DShape(input, weights, pad, stride, dilation, groups);
    }

    private IRType Visit(ITypeInferenceContext context, Conv2D target, TensorType input, TensorType weights)
    {
        var args = context.GetArguments(target, Conv2D.Stride, Conv2D.Padding, Conv2D.Dilation, Conv2D.Groups);
        return TypeInference.Conv2DType(input, weights, args[0], args[1], args[2], args[3]);
    }

    private IRType Visit(ITypeInferenceContext context, Conv2D target, DistributedType input, DistributedType weights, DistributedType bias)
    {
        if (Visit(context, target, input.TensorType, weights.TensorType) is not TensorType outType)
        {
            return new InvalidType(string.Empty);
        }

        var args = context.GetArguments(target, Conv2D.Stride, Conv2D.Padding, Conv2D.Dilation, Conv2D.Groups);

        // Not support split on h/w/r/s
        if (input.NdSBP.Any(sbp => sbp is SBPSplit s && s.Axis >= 2) || weights.NdSBP.Any(sbp => sbp is SBPSplit s && s.Axis >= 2))
        {
            return new InvalidType(string.Empty);
        }

        if (input.Placement != weights.Placement || input.Placement != bias.Placement)
        {
            return new InvalidType("placement not equal");
        }

        var ndsbp = new SBP[input.Placement.Rank];
        for (int i = 0; i < input.Placement.Rank; i++)
        {
            var invalid = new InvalidType($"({input.NdSBP[i]}, {weights.NdSBP[i]}) not support");
            switch (input.NdSBP[i], weights.NdSBP[i])
            {
                case (SBPSplit { Axis: int ax }, SBPSplit { Axis: int bx }):
                    // split on ic
                    if (ax == 1 && bx == 1)
                    {
                        if (bias.NdSBP[i] is SBPBroadCast)
                        {
                            ndsbp[i] = SBP.P;
                        }
                        else
                        {
                            return invalid;
                        }
                    }
                    else
                    {
                        return invalid;
                    }

                    break;
                case (SBPSplit { Axis: int ax }, SBPBroadCast):
                    if (ax == 0 && bias.NdSBP[i] is SBPBroadCast)
                    {
                        ndsbp[i] = SBP.S(ax);
                    }
                    else
                    {
                        return invalid;
                    }

                    break;
                case (SBPBroadCast, SBPSplit { Axis: int bx }):
                    if (bx == 0 && bias.NdSBP[i] is SBPSplit s && s.Axis == bx)
                    {
                        ndsbp[i] = SBP.S(bx + 1);
                    }
                    else
                    {
                        return invalid;
                    }

                    break;
                case (SBPBroadCast, SBPBroadCast):
                    if (bias.NdSBP[i] is SBPBroadCast)
                    {
                        ndsbp[i] = SBP.B;
                    }
                    else
                    {
                        return invalid;
                    }

                    break;
                default:
                    return invalid;
            }
        }

        return new DistributedType(outType, ndsbp, input.Placement);
    }
}
