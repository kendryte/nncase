// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Diagnostics;
using System.Linq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.Utilities;
using OrtKISharp;
using static Nncase.IR.F.Tensors;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.Utility;
using Reduce = Nncase.IR.Math.Reduce;

namespace Nncase.Evaluator.Math;

/// <summary>
/// Evaluator for <see cref="Reduce"/>.
/// </summary>
public class ReduceEvaluator : IEvaluator<Reduce>, ITypeInferencer<Reduce>, ICostEvaluator<Reduce>, IShapeEvaluator<Reduce>, IMetricEvaluator<Reduce>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, Reduce reduce)
    {
        var input = context.GetOrtArgumentValue(reduce, Reduce.Input);
        var axis = context.GetArgumentValueAsArray<long>(reduce, Reduce.Axis);
        var keepDims = context.GetArgumentValueAsScalar<long>(reduce, Reduce.KeepDims);

        // when HasBindedMixQuantInfo is true, eval will do simulation of quant/dequant for some inputs, this is used for evaluate accumulated quant error for layers.
        if (context.CurrentCall.EnodeBestQuantConfigWithCosine != null)
        {
            var pattern = IsRangeOfMarker(IsWildcard(), IsWildcard());
            if (pattern.MatchLeaf(context.CurrentCall.Arguments.ToArray()[0]) && ((Nncase.IR.Marker)context.CurrentCall.Arguments.ToArray()[0]).MixQuantInfo?.HasBindedMixQuantInfo == true)
            {
                var quantParam = ((Nncase.IR.Marker)context.CurrentCall.Arguments.ToArray()[0]).MixQuantInfo!.QuantParameter;

                // input feature map quantParam count should be 1 since input feature map quant is by tensor.
                Trace.Assert(quantParam.Count == 1);
                var inputFloat = input.ToArray<float>();
                for (var i = 0; i < inputFloat.Length; i++)
                {
                    var inputBufQuant = (double)((inputFloat[i] / (double)quantParam[0].Scale) + quantParam[0].ZeroPoint);
                    if (!(quantParam[0].Scale == 1.0f && quantParam[0].ZeroPoint == 0))
                    {
                        inputBufQuant = System.Math.Round((double)(float)inputBufQuant);
                    }

                    var inputBufDeQuant = (float)((inputBufQuant - quantParam[0].ZeroPoint) * (double)quantParam[0].Scale);
                    inputFloat[i] = (float)inputBufDeQuant;
                }

                input = OrtKISharp.Tensor.MakeTensor(inputFloat, input.Shape);
            }
        }

        return (reduce.ReduceOp switch
        {
            ReduceOp.Mean => OrtKI.ReduceMean(input, axis, keepDims),
            ReduceOp.Max => OrtKI.ReduceMax(input, axis, keepDims),
            ReduceOp.Min => OrtKI.ReduceMin(input, axis, keepDims),
            ReduceOp.Prod => OrtKI.ReduceProd(input, axis, keepDims),
            ReduceOp.Sum => OrtKI.ReduceSum(
                input,
                context.GetInt64OrtTensorArgumentValue(reduce, Reduce.Axis),
                keepDims,
                0),
            _ => throw new ArgumentOutOfRangeException(nameof(reduce)),
        }).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Reduce target)
    {
        var input = context.CheckArgumentType<TensorType>(target, Reduce.Input);
        context.CheckArgumentType<TensorType>(target, Reduce.Axis);
        return Visit(context, target, input);
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, Reduce target)
    {
        var input = context.GetArgumentType<TensorType>(target, Reduce.Input);
        var ret = context.GetReturnType<TensorType>();
        uint input_elem = input.Shape.Aggregate(1U, (acc, d) => acc * (d.IsFixed ? (uint)d.FixedValue : 1U));
        uint ret_elem = ret.Shape.Aggregate(1U, (acc, d) => acc * (d.IsFixed ? (uint)d.FixedValue : 1U));
        uint macPerElement = input_elem / ret_elem;
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(input),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(ret),
            [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(ret, macPerElement),
        };
    }

    public Expr Visit(IShapeEvaluateContext context, Reduce target)
    {
        var keepDims = context.GetArgument(target, Reduce.KeepDims);
        var axis = context.GetArgument(target, Reduce.Axis);
        if (keepDims is TensorConst keepDimsV &&
            axis is TensorConst axisValue)
        {
            var outShape = context.GetArgumentShape(target, Reduce.Input);
            var inShape = context.GetArgumentShape(target, Reduce.Input);
            var axes = axisValue.Value.Cast<int>();
            var keepDimsValue = keepDimsV.Value.ToScalar<int>();
            foreach (var axValue in axes)
            {
                var ax = ShapeExprUtility.Positive(axValue, inShape);
                if (keepDimsValue == 1)
                {
                    outShape = ShapeExprUtility.Replace(outShape, ax, 1);
                }
                else
                {
                    outShape = ShapeExprUtility.Remove(outShape, ax);
                }
            }

            return outShape;
        }

        throw new NotImplementedException();
    }

    public Metric Visit(IMetricEvaluateContext context, Reduce target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, Reduce.Input);
        var returnType = context.GetReturnType<TensorType>();
        var rF = MetricUtility.GetFLOPs(returnType);
        var iF = MetricUtility.GetFLOPs(inputType);
        var inner = iF / rF;
        _ = iF / inner;
        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(inputType) + CostUtility.GetMemoryAccess(returnType),
            [MetricFactorNames.FLOPs] = iF,
        };
    }

    private IRType Visit(ITypeInferenceContext context, Reduce target, TensorType input)
    {
        var args = context.GetArguments(target, Reduce.KeepDims, Reduce.Axis);
        return TypeInference.ReduceType(input, args[0], args[1]);
    }
}
