// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using DryIoc.FastExpressionCompiler.LightExpression;
using NetFabric.Hyperlinq;
using Nncase.CodeGen;
using Nncase.CostModel;
using Nncase.Evaluator.Math;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using Nncase.Utilities;
using OrtKISharp;
using static Nncase.IR.F.Math;
using static Nncase.IR.F.Tensors;
using Dimension = Nncase.IR.Dimension;
using Shape = Nncase.IR.Shape;
using Slice = Nncase.IR.Tensors.Slice;

namespace Nncase.Evaluator.Tensors;

/// <summary>
/// Evaluator for <see cref="Slice"/>.
/// </summary>
public class SliceEvaluator : IEvaluator<Slice>, ITypeInferencer<Slice>, ICostEvaluator<Slice>, IMetricEvaluator<Slice>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, Slice sl)
    {
        OrtKISharp.Tensor input;

        var inputOrg = context.GetArgumentValue(sl, Slice.Input).AsTensor();
        var dataType = inputOrg.ElementType;
        if (dataType.IsFloat() && dataType != DataTypes.Float32)
        {
            input = Cast(inputOrg, DataTypes.Float32).Evaluate().AsTensor().ToOrtTensor();
        }
        else
        {
            input = context.GetOrtArgumentValue(sl, Slice.Input);
        }

        var begins = context.GetInt64OrtTensorArgumentValue(sl, Slice.Begins);
        var ends = context.GetInt64OrtTensorArgumentValue(sl, Slice.Ends);
        var axes = context.GetInt64OrtTensorArgumentValue(sl, Slice.Axes);
        var strides = context.GetInt64OrtTensorArgumentValue(sl, Slice.Strides);
        var sliced = OrtKI.Slice(input, begins, ends, axes, strides);
        if (dataType.IsFloat() && dataType != DataTypes.Float32)
        {
            return Value.FromTensor(context.CurrentCall.CheckedType is AnyType ? sliced.ToTensor().CastTo(dataType) : sliced.ToTensor(context.CurrentCall.CheckedTensorType).CastTo(dataType));
        }
        else
        {
            return Value.FromTensor(context.CurrentCall.CheckedType is AnyType ? sliced.ToTensor() : sliced.ToTensor(context.CurrentCall.CheckedTensorType));
        }
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Slice target)
    {
        var input = context.CheckArgumentType<IRType>(target, Slice.Input);
        context.CheckArgumentTensorTypeOrBroadcast(target, Slice.Begins);
        context.CheckArgumentTensorTypeOrBroadcast(target, Slice.Ends);
        context.CheckArgumentTensorTypeOrBroadcast(target, Slice.Axes);
        context.CheckArgumentTensorTypeOrBroadcast(target, Slice.Strides);
        return input switch
        {
            TensorType t => Visit(context, target, t),
            DistributedType d => Visit(context, target, d),
            AnyType => AnyType.Default,
            _ => new InvalidType(input.GetType().Name),
        };
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, Slice target)
    {
        var outputType = context.GetReturnType<IRType>();

        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(outputType),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(outputType),
        };
    }

    public Metric Visit(IMetricEvaluateContext context, Slice target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, Slice.Input);
        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(inputType) * 2,
        };
    }

    /// <param name="axes">Axes.</param>
    /// <param name="input">Input type.</param>
    /// <param name="f">(index in axis, axis, inDim) -> outDim.</param>
    private static Shape ApplyAxis(int[] axes, TensorType input, Func<int, int, Dimension, Dimension> f)
    {
        if (input.Shape.IsUnranked)
        {
            return Shape.Unranked;
        }

        var outShape = input.Shape.ToArray();
        for (int i = 0; i < axes.Length; i++)
        {
            var axisV = axes[i];
            var axis = axisV < 0
                ? axisV + input.Shape.Rank
                : axisV;
            outShape[axis] = f(i, axis, input.Shape[axis]);
        }

        return outShape;
    }

    private static Dimension TranslateBeginEnd(Dimension x, Dimension dim, long lowerBound, long upperBoundBias)
    {
        if (x.IsFixed)
        {
            var newX = x.FixedValue < 0 ? dim + x : x;
            return Dimension.Clamp(newX, lowerBound, dim + upperBoundBias);
        }
        else
        {
            var newX = Select(x.Value < 0L, (dim + x).ToExpr(), x.ToExpr());
            return Dimension.Clamp(newX, lowerBound, dim + upperBoundBias);
        }
    }

    private IRType Visit(ITypeInferenceContext context, Slice target, TensorType input)
    {
        if (input.Shape.IsUnranked)
        {
            return input;
        }

        if (input.Shape.IsRanked && input.Shape.Count == 0)
        {
            return new InvalidType("Slice Input should not scalar");
        }

        var axes = ((TensorConst)context.GetDimensionArgument(target, Slice.Axes)).Value.ToArray<int>();
        var strides = ((TensorConst)context.GetDimensionArgument(target, Slice.Strides)).Value.ToArray<long>();
        var begins = context.GetDimensionArgument(target, Slice.Begins);
        var ends = context.GetDimensionArgument(target, Slice.Ends);
        if (begins.CheckedShape.IsFixed)
        {
            if (ends.CheckedShape.IsFixed)
            {
                if (begins.CheckedShape[0].FixedValue != ends.CheckedShape[0].FixedValue)
                {
                    return new InvalidType("Slice begins, ends, strides should be same length");
                }
            }

            if (begins.CheckedShape[0].FixedValue != strides.Length)
            {
                return new InvalidType("Slice begins, ends, strides should be same length");
            }
        }

        var outShape = ApplyAxis(axes, input, (i, axis, inDim) =>
        {
            var stride = strides[i];

            // reverse stride
            if (stride < 0)
            {
                // document in onnx operators:
                // for positive stepping and [0, dims[axes[i]]-1] for negative stepping.
                var begin = TranslateBeginEnd(begins[i], inDim, 0, -1);

                // while for negative stepping it is clamped to [-1, dims[axes[i]]-1].
                var end = TranslateBeginEnd(ends[i], inDim, -1, -1);
                return Dimension.CeilDiv(begin - end, Dimension.Abs(stride));
            }
            else
            {
                // starts[i] is clamped into the range [0, dims[axes[i]]]
                var begin = TranslateBeginEnd(begins[i], inDim, 0, 0);

                // end[i] is clamped into the range [0, dims[axes[i]]]
                var end = TranslateBeginEnd(ends[i], inDim, 0, 0);
                return Dimension.CeilDiv(end - begin, Dimension.Abs(stride));
            }
        });
        return input with { Shape = outShape };
    }

    private IRType Visit(ITypeInferenceContext context, Slice target, DistributedType input)
    {
        var outType = Visit(context, target, input.TensorType);
        if (outType is not TensorType tensorType)
        {
            return new InvalidType("not support input tensor type infer");
        }

        var axes = ((TensorConst)context.GetArgument(target, Slice.Axes)).Value.ToArray<int>();
        if (Enumerable.Range(0, input.AxisPolices.Count).Any(i => input.AxisPolices[i] is SBPSplit && axes.Contains(i)))
        {
            return new InvalidType("not support input tensor type infer");
        }

        return new DistributedType((TensorType)outType, input.AxisPolices, input.Placement);
    }
}
