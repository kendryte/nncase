// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.NN;
using TorchSharp;
using torchF = TorchSharp.torch.nn.functional;

namespace Nncase.Evaluator.NN;

/// <summary>
/// Evaluator for <see cref="SpaceToBatch"/>.
/// </summary>
public class SpaceToBatchEvaluator : IEvaluator<SpaceToBatch>, ITypeInferencer<SpaceToBatch>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, SpaceToBatch conv)
    {
        throw new NotImplementedException();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, SpaceToBatch target)
    {
        var input = context.CheckArgumentType<TensorType>(target, SpaceToBatch.Input);
        var blockShape = context.CheckArgumentType<TensorType>(target, SpaceToBatch.BlockShape);
        var paddings = context.CheckArgumentType<TensorType>(target, SpaceToBatch.Paddings);
        return Visit(context, target, input, blockShape, paddings);
    }

    private IRType Visit(ITypeInferenceContext context, SpaceToBatch target, TensorType input, TensorType blockShape, TensorType paddings)
    {
        if (context.GetArgument(target, SpaceToBatch.BlockShape) is TensorConst block_shape_con &&
             context.GetArgument(target, SpaceToBatch.Paddings) is TensorConst paddings_con)
        {
            var ts_block_shape = block_shape_con.Value.Cast<int>();
            var ts_paddings = paddings_con.Value.Cast<int>();
            int m = (int)ts_block_shape.Length;
            var padded_shape = input.Shape.ToList();
            for (int i = 1; i < 1 + m; i++)
            {
                if (!padded_shape[i].IsUnknown)
                {
                    padded_shape[i] += new Dimension(ts_paddings[i, 0] + ts_paddings[i, 1]);
                }
            }

            var outshape = new List<Dimension> { padded_shape[0] };
            foreach (var i in Enumerable.Range(1, m))
            {
                outshape.Add(padded_shape[i].IsUnknown ? Dimension.Unknown :
                                    padded_shape[i].FixedValue % ts_block_shape[i - 1] == 0 ?
                                      padded_shape[i].FixedValue / ts_block_shape[i - 1] :
                                      throw new TypeInferenceInterruptException(
                                        new InvalidType($"The Padded Shape Must Divides BlockShape!")
                                      ));
            }

            foreach (var i in Enumerable.Range(m + 1, outshape.Count - (m + 1)))
            {
                outshape.Add(padded_shape[i]);
            }

            foreach (var block in ts_block_shape)
            {
                outshape[0] = outshape[0].IsUnknown ? Dimension.Unknown : outshape[0].FixedValue * block;
            }

            return input with { Shape = new Shape(outshape) };
        }

        return new InvalidType("Can't Infer Shape With Dynamic Input!");
    }
}
