// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Linq;
using Nncase.IR;
using Nncase.IR.Tensors;
using OrtKISharp;

namespace Nncase.Evaluator.Tensors;

/// <summary>
/// Evaluator for <see cref="Prod"/>.
/// </summary>
public class ProdEvaluator : IEvaluator<Prod>, ITypeInferencer<Prod>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, Prod prod)
    {
        var input = context.GetOrtArgumentValue(prod, Prod.Input);
        return OrtKI.ReduceProd(
            input,
            Enumerable.Range(0, input.Shape.Length).Select(x => (long)x).ToArray(),
            0).ToValue();
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
