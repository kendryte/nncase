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
public class UnsqueezeEvaluator : IEvaluator<Unsqueeze>, ITypeInferencer<Unsqueeze>, ICostEvaluator<Unsqueeze>, IShapeEvaluator<Unsqueeze>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, Unsqueeze unSqueeze)
    {
        var input = context.GetOrtArgumentValue(unSqueeze, Unsqueeze.Input);
        var axes = context.GetInt64OrtTensorArgumentValue(unSqueeze, Unsqueeze.Dim);
        return OrtKI.Unsqueeze(input, axes).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Unsqueeze target)
    {
        var input = context.CheckArgumentType<TensorType>(target, Unsqueeze.Input);
        _ = context.CheckArgumentType<TensorType>(target, Unsqueeze.Dim);
        return Visit(context, target, input);
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, Unsqueeze target)
    {
        return new()
        {
            [CostFactorNames.CPUCycles] = 1,
        };
    }

    private IRType Visit(ITypeInferenceContext context, Unsqueeze target, TensorType input)
    {
        if (input.Shape.IsUnranked)
        {
            return input;
        }

        if (context.GetArgument(target, Unsqueeze.Dim) is TensorConst tdims)
        {
            var dimsValue = tdims.Value.Cast<int>();
            var outShape = input.Shape.ToList();
            foreach (var dimVal in dimsValue)
            {
                if (dimVal >= 0)
                {
                    outShape.Insert(dimVal, 1);
                }
                else
                {
                    var index = System.Math.Max(outShape.Count + dimVal + 1, 0);
                    outShape.Insert(index, 1);

                    // count == 3, dimVal == -4
                }
            }

            return input with { Shape = new Shape(outShape) };
        }

        return input with { Shape = new Shape(Enumerable.Repeat(Dimension.Unknown, input.Shape.Rank + 1)) };
    }

    public Expr Visit(IShapeEvaluateContext context, Unsqueeze target)
    {
        var dims = context.GetArgument(target, Unsqueeze.Dim);
        if (dims is TensorConst dimsConst)
        {
            var dimsValue = dimsConst.Value.ToArray<int>();
            var outShape = context.GetArgumentShape(target, Unsqueeze.Input);

            foreach (var dimVal in dimsValue)
            {
                if (dimVal >= 0)
                {
                    outShape = ShapeExprUtility.Insert(outShape, dimVal, 1);
                }
                else
                {
                    var index = IR.F.Math.Max(IR.F.Tensors.ShapeOf(outShape)[0] + dimVal + 1, 0);
                    outShape = ShapeExprUtility.Insert(outShape, index, 1);
                }
            }

            return outShape;
        }

        throw new NotImplementedException();
    }
}
