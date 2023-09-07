// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using DryIoc.ImTools;
using NetFabric.Hyperlinq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Tensors;
using Nncase.Utilities;
using OrtKISharp;
using static Nncase.Evaluator.TypeInference;
using static Nncase.IR.F.Math;
using static Nncase.IR.F.Tensors;
using Range = Nncase.IR.Tensors.Range;
using Reshape = Nncase.IR.Tensors.Reshape;

namespace Nncase.Evaluator.Tensors;

/// <summary>
/// Evaluator for <see cref="Range"/>.
/// </summary>
public class ReshapeEvaluator : IEvaluator<Reshape>, ITypeInferencer<Reshape>, ICostEvaluator<Reshape>, IShapeEvaluator<Reshape>, IMetricEvaluator<Reshape>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, Reshape reshape)
    {
        var input = context.GetOrtArgumentValue(reshape, Reshape.Input);
        var shape = context.GetInt64OrtTensorArgumentValue(reshape, Reshape.Shape);
        return OrtKI.Reshape(input, shape, context.CurrentCall.CheckedType is TensorType && context.CurrentCall.CheckedShape.IsFixed ? (context.CurrentCall.CheckedShape.ToValueArray().Contains(0) ? 1 : 0) : 0).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Reshape target)
    {
        var input = context.CheckArgumentType<TensorType>(target, Reshape.Input);
        return Visit(context, target, input);
    }

    public Cost Visit(ICostEvaluateContext context, Reshape target)
    {
        return CostUtility.GetReshapeCost();
    }

    Cost ICostEvaluator<Reshape>.Visit(ICostEvaluateContext context, Reshape target)
    {
        return new()
        {
            [CostFactorNames.CPUCycles] = 1,
        };
    }

    public Expr Visit(IShapeEvaluateContext context, Reshape target)
    {
        var inShape = context.GetArgumentShape(target, Reshape.Input);
        var shape = context.GetArgument(target, Reshape.Shape);
        return IR.F.ShapeExpr.ReshapeShape(inShape, shape);
    }

    public Metric Visit(IMetricEvaluateContext context, Reshape target)
    {
        return Metric.Zero;
    }

    private IRType Visit(ITypeInferenceContext context, Reshape target, TensorType input)
    {
        if (input.Shape.IsUnranked)
        {
            return input;
        }

        if (context.GetArgument(target, Reshape.Shape) is TensorConst shapeConst)
        {

            var shapeValue = shapeConst.Value.ToArray<int>();
            var negCount = shapeValue.Count(IsMinus1);
            var inputSize = input.Shape.Prod().FixedValue;
            var shapeSize = shapeValue.Aggregate(1, (x, y) => x * y);
            if (negCount > 1)
            {
                return new InvalidType(
                    $"Reshape at most one dimension of the new shape can be -1," +
                    $" shape:{shapeValue}");
            }

            if (input.Shape.IsFixed)
            {
                if (negCount < 1)
                {

                    if (inputSize != shapeSize)
                    {
                        return new InvalidType("Reshape input shape size and param shape size must be same," +
                                               $" shape:{shapeValue.ToArray().Aggregate(string.Empty, (s, i) => s + i + " ")}, input shape${string.Join(",", input.Shape)}");
                    }

                    return input with { Shape = new Shape(shapeValue) };
                }
                else
                {
                    shapeSize = -shapeSize;
                    var negIndex = shapeValue.Select((dim, index) => (dim, index)).First(x => IsMinus1(x.dim)).index;
                    if (inputSize % shapeSize != 0)
                    {
                        return new InvalidType("Reshape input size must be divisible by shapeSize when has -1");
                    }

                    shapeValue[negIndex] = inputSize / shapeSize;
                    return input with { Shape = new Shape(shapeValue) };
                }
            }
            else
            {
                return input with
                {
                    Shape = new Shape(shapeValue.Select(x => x == -1 ? Dimension.Unknown : x).ToArray()),
                };
            }
        }

        var targetType = context.CheckArgumentType<TensorType>(target, Reshape.Shape);
        var outShape = ReshapeTo(targetType);
        return input with { Shape = outShape };
    }
}
