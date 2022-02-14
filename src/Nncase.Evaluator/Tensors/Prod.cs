// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Linq;
using Nncase.IR;
using Nncase.IR.Tensors;

namespace Nncase.Evaluator.Tensors;

/// <summary>
/// Evaluator for <see cref="Prod"/>.
/// </summary>
public class ProdEvaluator : IEvaluator<Prod>, ITypeInferencer<Prod>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, Prod prod)
    {
        var input = context.GetTorchArgumentValue(prod, Prod.Input);
        var size = input.shape.Aggregate(1L, (sum, v) => sum * v);
        var v = input.reshape(size).cumprod(0)[size - 1];
        return v.ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Prod target)
    {
        var input = context.CheckArgumentType<TensorType>(target, Prod.Input);
        return Visit(context, target, input);
    }

    private IRType Visit(ITypeInferenceContext context, Prod target, TensorType input)
    {
        return new TensorType(input.DType, Shape.Scalar);
    }
}
