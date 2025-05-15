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
        var inputTensor = context.GetArgumentValueAsTensor(sl, Slice.Input);
        var inputElemType = inputTensor.ElementType;
        if (TryGetTempDataType(inputTensor.ElementType, out var tempType))
        {
            input = inputTensor.CastTo(tempType).ToOrtTensor();
        }
        else
        {
            input = inputTensor.ToOrtTensor();
        }

        var begins = context.GetInt64OrtTensorArgumentValue(sl, Slice.Begins);
        var ends = context.GetInt64OrtTensorArgumentValue(sl, Slice.Ends);
        var axes = context.GetInt64OrtTensorArgumentValue(sl, Slice.Axes);
        var strides = context.GetInt64OrtTensorArgumentValue(sl, Slice.Strides);
        var sliced = OrtKI.Slice(input, begins, ends, axes, strides);

        switch (context.CurrentCall.CheckedType)
        {
            case AnyType:
                {
                    var slicedTensor = sliced.ToTensor();
                    if (tempType is not null)
                    {
                        slicedTensor = slicedTensor.CastTo(inputElemType);
                    }

                    return Value.FromTensor(slicedTensor);
                }

            case TensorType tensorType:
                {
                    Tensor slicedTensor;
                    if (tempType is not null)
                    {
                        slicedTensor = sliced.ToTensor(tensorType with { DType = tempType }).CastTo(inputElemType);
                    }
                    else
                    {
                        slicedTensor = sliced.ToTensor(tensorType);
                    }

                    return Value.FromTensor(slicedTensor);
                }

            default:
                throw new NotSupportedException("Unsupported type.");
        }
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Slice target)
    {
        var input = context.CheckArgumentType<IRType>(target, Slice.Input);
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
    private static Shape ApplyAxis(long[] axes, TensorType input, Func<int, long, Dimension, Dimension> f)
    {
        if (input.Shape is not RankedShape inShape)
        {
            return Shape.Unranked;
        }

        var outShape = inShape.ToArray();
        for (int i = 0; i < axes.Length; i++)
        {
            var axisV = axes[i];
            var axis = axisV < 0
                ? axisV + inShape.Rank
                : axisV;
            outShape[axis] = f(i, axis, inShape[axis]);
        }

        return outShape;
    }

    private static Dimension TranslateBeginEnd(Dimension x, Dimension dim, long lowerBound, long upperBoundBias)
    {
        var newX = Dimension.Positive(x, dim);
        return Dimension.Clamp(newX, lowerBound, dim + upperBoundBias);
    }

    private IRType Visit(ITypeInferenceContext context, Slice target, TensorType input)
    {
        if (input.Shape.IsUnranked)
        {
            return input;
        }

        if (input.Shape.IsScalar)
        {
            return new InvalidType("Slice Input should not scalar");
        }

        var axes = ((Shape)context.GetArgument(target, Slice.Axes)).ToValueArray();
        var strides = ((Shape)context.GetArgument(target, Slice.Strides)).ToValueArray();
        var begins = (Shape)context.GetArgument(target, Slice.Begins);
        var ends = (Shape)context.GetArgument(target, Slice.Ends);
        if (begins.IsRanked)
        {
            if (ends.IsRanked)
            {
                if (begins.Rank != ends.Rank)
                {
                    return new InvalidType("Slice begins, ends, strides should be same length");
                }
            }

            if (begins.Rank != strides.Length)
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

        var axes = ((Shape)context.GetArgument(target, Slice.Axes)).ToValueArray();
        if (Enumerable.Range(0, input.AxisPolices.Count).Any(i => input.AxisPolices[i] is SBPSplit && axes.Contains(i)))
        {
            return new InvalidType("not support input tensor type infer");
        }

        return new DistributedType((TensorType)outType, input.AxisPolices, input.Placement);
    }

    private bool TryGetTempDataType(DataType dataType, out DataType tempType)
    {
        tempType = null!;
        bool change = false;

        switch (dataType)
        {
            case PrimType pt when pt == DataTypes.Float8E4M3 || pt == DataTypes.Float8E5M2:
                tempType = DataTypes.Float32;
                change = true;
                break;
            case VectorType vtype:
                if (TryGetTempDataType(vtype.ElemType, out var tempElemType))
                {
                    tempType = vtype with { ElemType = tempElemType };
                    change = true;
                }

                break;
            default:
                break;
        }

        return change;
    }
}
