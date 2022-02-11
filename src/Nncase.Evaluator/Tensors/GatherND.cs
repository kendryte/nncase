// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Linq;
using Nncase.IR;
using Nncase.IR.Tensors;
using Tensorflow;
using static Tensorflow.Binding;

namespace Nncase.Evaluator.Tensors;

/// <summary>
/// Evaluator for <see cref="GatherND"/>.
/// </summary>
public class GatherNDEvaluator : IEvaluator<GatherND>, ITypeInferencer<GatherND>
{
    /// <inheritdoc/>
    public Const Visit(IEvaluateContext context, GatherND gatherND)
    {
        var input = context.GetTFArgumentValue(gatherND, GatherND.Input);
        var indices = context.GetTFArgumentValue(gatherND, GatherND.Index);
        var batchDims = context.GetTFArgumentValue(gatherND, GatherND.BatchDims);
        return GatherNDImpl(input, indices, batchDims).ToConst();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, GatherND target)
    {
        var input = context.CheckArgumentType<TensorType>(target, GatherND.Input);
        var batchDims = context.CheckArgumentType<TensorType>(target, GatherND.BatchDims);
        var index = context.CheckArgumentType<TensorType>(target, GatherND.Index);
        return Visit(context, target, input, batchDims, index);
    }

    private IRType Visit(ITypeInferenceContext context, GatherND target, TensorType input, TensorType batchDims, TensorType index)
    {
        if (context.GetArgument(target, GatherND.BatchDims) is Const batchDimsValue)
        {
            var lastIndexDims = index.Shape.Last();
            if (!lastIndexDims.IsFixed)
            {
                return new InvalidType("GatherND input last dim is dynamic, can't infer result shape");
            }

            // result shape = index_shape[:-1] + input_shape[index_shape[-1] + batch_dims:]
            var dimensions = index.Shape.ToArray()[..(index.Shape.Rank - 1)];
            var d = lastIndexDims.FixedValue + batchDimsValue.ToScalar<int>();
            var shapeValue = dimensions.Concat(input.Shape.ToArray()[d..]);
            return new TensorType(input.DType, new IR.Shape(shapeValue));
        }
        else
        {
            return new InvalidType("GatherND batch_dims must be constant");
        }
    }

    private Tensor GatherNDImpl(Tensor input, Tensor indices, Tensor batchDims)
    {
        return tf.Context.ExecuteOp(
            "GatherNd",
            null!,
            new ExecuteOpArgs(input, indices));
    }
}
