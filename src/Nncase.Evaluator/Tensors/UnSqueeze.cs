// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Linq;
using Nncase.IR;
using Nncase.IR.Tensors;
using OrtKISharp;

namespace Nncase.Evaluator.Tensors;

/// <summary>
/// Evaluator for <see cref="Unsqueeze"/>.
/// </summary>
public class UnsqueezeEvaluator : IEvaluator<Unsqueeze>, ITypeInferencer<Unsqueeze>
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
        return Visit(context, target, input);
    }

    private IRType Visit(ITypeInferenceContext context, Unsqueeze target, TensorType input)
    {
        if (context.GetArgument(target, Unsqueeze.Dim) is TensorConst tdims)
        {
            var dimsValue = tdims.Value.Cast<int>();
            var outShape = input.Shape.ToList();
            foreach (var dimVal in dimsValue)
            {
                if (dimVal > 0)
                {
                    outShape.Insert(dimVal, 1);
                }
                else
                {
                    // count == 3, dimVal == -4
                    var index = System.Math.Max(outShape.Count + dimVal, 0);
                    outShape.Insert(index, 1);
                }
            }

            return input with { Shape = new Shape(outShape) };
        }

        return input with { Shape = new Shape(Enumerable.Repeat(Dimension.Unknown, input.Shape.Rank + 1)) };
    }
}
