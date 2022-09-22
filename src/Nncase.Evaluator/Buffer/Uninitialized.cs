// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Linq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Buffer;
using OrtKISharp;

namespace Nncase.Evaluator.Buffer;

/// <summary>
/// Evaluator for <see cref="Uninitialized"/>.
/// </summary>
public class UninitializedEvaluator : ITypeInferencer<Uninitialized>, IEvaluator<Uninitialized>, ICostEvaluator<Uninitialized>
{
    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Uninitialized target)
    {
        if (context.GetArgument(target, Uninitialized.Shape) is TensorConst tensor)
        {
            return new TensorType(target.DType, tensor.Value.ToArray<int>());
        }
        var shape = context.CheckArgumentType<TensorType>(target, Uninitialized.Shape);
        return new TensorType(target.DType, new(Enumerable.Repeat(Dimension.Unknown, shape.Shape[0].FixedValue)));
    }

    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, Uninitialized target)
    {
        var shape = context.GetArgumentValueAsArray<int>(target, Uninitialized.Shape);
        return Value.FromTensor(Tensor.FromBytes(target.DType,
          Enumerable.Repeat<byte>(0, (int)TensorUtilities.GetProduct(shape) * target.DType.SizeInBytes).ToArray(),
          shape));
    }

    /// <inheritdoc/>
    public Cost? Visit(ICostEvaluateContext context, Uninitialized target)
    {
        var outputType = context.GetReturnType<TensorType>();
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(outputType),
        };
    }

}
