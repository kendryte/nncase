// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Tensors;
using Nncase.Utilities;
using OrtKISharp;

namespace Nncase.Evaluator.Tensors;

/// <summary>
/// Evaluator for <see cref="Unsqueeze"/>.
/// </summary>
public class UnsqueezeEvaluator : IEvaluator<Unsqueeze>, ITypeInferencer<Unsqueeze>, ICostEvaluator<Unsqueeze>, IShapeEvaluator<Unsqueeze>, IMetricEvaluator<Unsqueeze>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, Unsqueeze unSqueeze)
    {
        var inputValue = context.GetArgumentValue(unSqueeze, Unsqueeze.Input);
        var axesValue = context.GetArgumentValue(unSqueeze, Unsqueeze.Dim);

        switch (inputValue, axesValue)
        {
            case (_, TensorValue axesTValue):
                var axesTensor = axesTValue.AsTensor();
                if (inputValue is TensorValue inputTValue)
                {
                    var input = inputTValue.AsTensor().ToOrtTensor();
                    var axes = axesTensor.Cast<long>().ToOrtTensor();
                    return Value.FromTensor(OrtKI.Unsqueeze(input, axes).ToTensor(context.CurrentCall.CheckedTensorType));
                }
                else if (inputValue is DimensionValue inputDValue)
                {
                    if (axesTensor.Shape.Rank > 1 || axesTensor.ToScalar<int>() != 0)
                    {
                        throw new NotSupportedException("only support scalar dim when input is DimensionValue!");
                    }

                    return new ShapeValue(new[] { inputDValue });
                }

                break;
            default:
                break;
        }

        throw new NotSupportedException();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Unsqueeze target)
    {
        var input = context.CheckArgumentType<IRType>(target, Unsqueeze.Input);
        if (input is TensorType tensorType)
        {
            return Visit(context, target, tensorType);
        }
        else if (input is DistributedType distributedType)
        {
            return Visit(context, target, distributedType);
        }

        return new InvalidType(input.GetType().Name);
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, Unsqueeze target)
    {
        return new()
        {
            [CostFactorNames.CPUCycles] = 1,
        };
    }

    public Expr Visit(IShapeEvaluateContext context, Unsqueeze target)
    {
        var input = context.GetArgumentShape(target, Unsqueeze.Input);
        var dims = context.GetArgument(target, Unsqueeze.Dim);
        return IR.F.ShapeExpr.UnsqueezeShape(input, dims);
    }

    public Metric Visit(IMetricEvaluateContext context, Unsqueeze target) => Metric.Zero;

    private IRType Visit(ITypeInferenceContext context, Unsqueeze target, TensorType input)
    {
        if (input.Shape.IsUnranked)
        {
            return input;
        }

        if (context.GetDimensionArgument(target, Unsqueeze.Dim) is TensorConst axes)
        {
            var axesValue = axes.Value.ToArray<int>();
            var outShape = new Dimension[input.Shape.Rank + axesValue.Length];
            axesValue = axesValue.Select(axis => axis < 0 ? axis + outShape.Length : axis).ToArray();
            var offset = 0;
            for (int i = 0; i < outShape.Length; i++)
            {
                if (axesValue.Contains(i))
                {
                    outShape[i] = 1;
                }
                else
                {
                    outShape[i] = input.Shape[offset++];
                }
            }

            return input with { Shape = new Shape(outShape) };
        }

        return input with { Shape = Shape.Unknown(input.Shape.Rank + 1) };
    }

    private IRType Visit(ITypeInferenceContext context, Unsqueeze target, DistributedType input)
    {
        var tensorType = (TensorType)Visit(context, target, input.TensorType);

        var ndsbp = new SBP[input.NdSBP.Count];

        if (context.GetArgument(target, Unsqueeze.Dim) is TensorConst tdims)
        {
            var dimsValue = tdims.Value.Cast<int>();
            for (int i = 0; i < input.NdSBP.Count; i++)
            {
                ndsbp[i] = input.NdSBP[i] switch
                {
                    SBPSplit { Axis: int axis } => SBP.S(axis + dimsValue.Select(i => i <= axis).Count(b => b)),
                    SBP sbp => sbp,
                };
            }
        }

        return new DistributedType(tensorType, ndsbp, input.Placement);
    }
}
