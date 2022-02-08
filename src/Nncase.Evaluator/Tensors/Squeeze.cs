// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Linq;
using Nncase.IR;
using Nncase.IR.Tensors;
using static Tensorflow.Binding;
using torchF = TorchSharp.torch.nn.functional;

namespace Nncase.Evaluator.Tensors;

/// <summary>
/// Evaluator for <see cref="Squeeze"/>.
/// </summary>
public class SqueezeEvaluator : IEvaluator<Squeeze>, ITypeInferencer<Squeeze>
{
    /// <inheritdoc/>
    public Const Visit(EvaluatorContext context, Squeeze squeeze)
    {
        var input = context.GetTFArgument(squeeze, Squeeze.Input);
        var dims = context.GetArgumentConst(squeeze, Squeeze.Dim).ToArray<int>();
        return tf.squeeze(input, dims).ToConst();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Squeeze target)
    {
        var input = context.CheckArgumentType<TensorType>(target, Split.Input);
        return Visit(context, target, input);
    }

    private IRType Visit(ITypeInferenceContext context, Squeeze target, TensorType input)
    {
        if (context.GetArgument(target, Squeeze.Dim) is Const dim_con)
        {
            var dims = dim_con.ToArray<int>();
            var outshape = input.Shape.ToList();
            foreach (var dimValue in dims)
            {
                if (outshape[dimValue].IsFixed && outshape[dimValue] == 1)
                {
                    outshape[dimValue] = int.MaxValue;
                }
                else
                {
                    return new InvalidType("The Shape[dim] is not 1!");
                }
            }

            return input with { Shape = new Shape(outshape.Where(x => x != int.MaxValue)) };
        }

        return input with { Shape = new Shape(Enumerable.Repeat(Dimension.Unknown, input.Shape.Count() - 1)) };
    }
}
