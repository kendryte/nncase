// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Linq;
using Nncase.IR;
using Nncase.IR.Tensors;
using static Tensorflow.Binding;

namespace Nncase.Evaluator.Tensors;

/// <summary>
/// Evaluator for <see cref="Unsqueeze"/>.
/// </summary>
public class UnsqueezeEvaluator : IEvaluator<Unsqueeze>, ITypeInferencer<Unsqueeze>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, Unsqueeze unSqueeze)
    {
        var input = context.GetTFArgumentValue(unSqueeze, Unsqueeze.Input);
        var dims = context.GetArgumentValueAsTensor<int>(unSqueeze, Unsqueeze.Dim)
            .Select(
                x => Util.PositiveIndex(x, input.shape.rank))
            .ToArray();
        foreach (var dim in dims)
        {
            input = tf.expand_dims(input, Util.PositiveIndex(dim, input.shape.rank));
        }

        return input.ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Unsqueeze target)
    {
        var input = context.CheckArgumentType<TensorType>(target, Split.Input);
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
                var dimV = Util.PositiveIndex(dimVal, input);
                if (dimV < 0)
                {
                    for (int i = dimV; i < 0; i++)
                    {
                        outShape.Insert(0, 1);
                    }
                }
            }

            return input with { Shape = new Shape(outShape) };
        }

        return input with { Shape = new Shape(Enumerable.Repeat(Dimension.Unknown, input.Shape.Rank + 1)) };
    }
}
