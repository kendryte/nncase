// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Linq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Buffers;
using OrtKISharp;

namespace Nncase.Evaluator.Buffers;

/// <summary>
/// Evaluator for <see cref="Uninitialized"/>.
/// </summary>
public class UninitializedEvaluator : IEvaluator<Uninitialized>, ITypeInferencer<Uninitialized>, ICostEvaluator<Uninitialized>
{
    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Uninitialized target)
    {
        TensorType tensorType;
        if (context.GetArgument(target, Uninitialized.Shape) is TensorConst tensor)
        {
            tensorType = new TensorType(target.DType, tensor.Value.ToArray<int>());
        }
        else
        {
            var shape = context.CheckArgumentType<TensorType>(target, Uninitialized.Shape);
            tensorType = new TensorType(target.DType, new(Enumerable.Repeat(Dimension.Unknown, shape.Shape[0].FixedValue)));
        }

        return target.Placement.Rank == 0 ? tensorType : new DistributedType(tensorType, target.NdSBP, target.Placement);
    }

    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, Uninitialized target)
    {
        return Value.None;
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, Uninitialized target)
    {
        return Cost.Zero;
    }
}
