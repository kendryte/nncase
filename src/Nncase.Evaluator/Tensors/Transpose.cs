// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Tensors;
using OrtKISharp;
using Tensorflow;
using Tensorflow.NumPy;
using static Tensorflow.Binding;

namespace Nncase.Evaluator.Tensors;

/// <summary>
/// Evaluator for <see cref="Transpose"/>.
/// </summary>
public class TransposeEvaluator : IEvaluator<Transpose>, ITypeInferencer<Transpose>, ICostEvaluator<Transpose>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, Transpose tr)
    {
        var input = context.GetOrtArgumentValue(tr, Transpose.Input);
        var perm = context.GetArgumentValueAsArray<long>(tr, Transpose.Perm);
        return OrtKI.Transpose(input, perm).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Transpose target)
    {
        var input = context.CheckArgumentType<TensorType>(target, Transpose.Input);
        return Visit(context, target, input);
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, Transpose target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, Transpose.Input);
        var outputType = context.GetReturnType<TensorType>();

        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(inputType),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(outputType),
        };
    }

    private IRType Visit(ITypeInferenceContext context, Transpose target, TensorType input)
    {
        var permExpr = context.GetArgument(target, Transpose.Perm);
        return TypeInference.TransposeType(input, permExpr);
    }
}
