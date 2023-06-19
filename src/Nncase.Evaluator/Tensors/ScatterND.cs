// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Linq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Tensors;
using OrtKISharp;

namespace Nncase.Evaluator.Tensors;

/// <summary>
/// Evaluator for <see cref="ScatterND"/>.
/// </summary>
public class ScatterNDEvaluator : IEvaluator<ScatterND>, ITypeInferencer<ScatterND>, ICostEvaluator<ScatterND>, IShapeEvaluator<ScatterND>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, ScatterND target)
    {
#if false
        var input = context.GetOrtArgumentValue(target, ScatterND.Input);
        var indices = context.GetInt64OrtTensorArgumentValue(target, ScatterND.Indices);
        var updates = context.GetOrtArgumentValue(target, ScatterND.Updates);

        return OrtKI.ScatterND(input, indices, updates, reduction: "none").ToValue();
#else
        var input = context.GetArgumentValueAsTensor(target, ScatterND.Input);
        var indices = context.GetArgumentValueAsTensor<int>(target, ScatterND.Indices);
        var updates = context.GetArgumentValueAsTensor(target, ScatterND.Updates);
        var update_indices = indices.Shape.ToValueArray().Take(0..(indices.Shape.Rank - 1)).Select(i => Enumerable.Range(0, i));
        var output = Tensor.FromBytes(input.ElementType, input.BytesBuffer.ToArray(), input.Shape);
        var indicesSpan = indices.Buffer.Span;
        var updatesSpan = updates.BytesBuffer;
        var updatesRemain = updates.Shape.ToValueArray().Skip(indices.Rank - 1).TakeOrDefault(updates.Shape.Rank - indices.Rank + 1, 1);
        var updateSize = updatesRemain.Any() ? updatesRemain.Aggregate((x, y) => x * y) * input.ElementType.SizeInBytes : input.ElementType.SizeInBytes;
        var outputSpan = output.BytesBuffer;
        var indicesSpanStride = indices.Strides.ToArray().Take(0..(indices.Shape.Rank - 1)).ToArray();
        var updatesSliceStride = updates.Strides.ToArray().SkipLast(updatesRemain.Count()).Select(s => s * input.ElementType.SizeInBytes).ToArray();
        var outputSpanStride = output.Strides.ToArray().SkipLast(updatesRemain.Count()).Select(s => s * input.ElementType.SizeInBytes).ToArray();
        foreach (var idx in LinqExtensions.CartesianProduct(update_indices))
        {
            var index = indicesSpan.Slice(TensorUtilities.GetIndex(indicesSpanStride, idx.ToArray()), indices.Shape.ToValueArray()[^1]);
            var updatesSlice = updatesSpan.Slice(TensorUtilities.GetIndex(updatesSliceStride, idx.ToArray()), updateSize);
            updatesSlice.CopyTo(outputSpan.Slice(TensorUtilities.GetIndex(outputSpanStride, index.ToArray())));
        }

        return Value.FromTensor(output);
#endif
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, ScatterND target)
    {
        var input = context.CheckArgumentType<TensorType>(target, ScatterND.Input);
        return input;
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, ScatterND target)
    {
        var returnType = context.GetReturnType<IRType>();

        return new()
        {
            [CostFactorNames.MemoryLoad] = returnType switch
            {
                TensorType t => CostUtility.GetMemoryAccess(t),
                _ => 1,
            },
            [CostFactorNames.MemoryStore] = returnType switch
            {
                TensorType t => CostUtility.GetMemoryAccess(t),
                _ => 1,
            },
        };
    }

    public Expr Visit(IShapeEvaluateContext context, ScatterND target) => context.GetArgumentShape(target, ScatterND.Input);
}
