// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Linq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Tensors;
using OrtKISharp;

namespace Nncase.Evaluator.Tensors;

/// <summary>
/// Evaluator for <see cref="GatherElements"/>.
/// </summary>
public class GatherElementsEvaluator : IEvaluator<GatherElements>, ITypeInferencer<GatherElements>, ICostEvaluator<GatherElements>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, GatherElements target)
    {
        var input = context.GetOrtArgumentValue(target, GatherElements.Input);
        var indices = context.GetInt64OrtTensorArgumentValue(target, GatherElements.Indices);
        var axis = context.GetArgumentValueAsScalar<long>(target, GatherElements.Axis);

        return OrtKI.GatherElements(input, indices, axis).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, GatherElements target)
    {
        var input = context.CheckArgumentType<TensorType>(target, GatherElements.Input);
		var indices = context.CheckArgumentType<TensorType>(target, GatherElements.Indices);
		
        return new TensorType(input.DType, indices.Shape);
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, GatherElements target)
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
}
