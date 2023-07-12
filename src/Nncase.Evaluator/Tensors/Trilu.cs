// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Tensors;
using Nncase.Utilities;
using OrtKISharp;

namespace Nncase.Evaluator.Tensors;

/// <summary>
/// Evaluator for <see cref="Trilu"/>.
/// </summary>
[EvaluatorGenerator]
[TypeInferGenerator]
public class TriluEvaluator : IEvaluator<Trilu>, ITypeInferencer<Trilu>, ICostEvaluator<Trilu>, IShapeEvaluator<Trilu>
{
    /// <inheritdoc/>
    public IValue Visit(OrtKISharp.Tensor input, OrtKISharp.Tensor k, long upper)
    {
        return OrtKI.Trilu(input, k, upper).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(TensorType input)
    {
        return input;
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, Trilu target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, Transpose.Input);
        var outputType = context.GetReturnType<TensorType>();

        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(inputType),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(outputType),
        };
    }

    public Expr Visit(IShapeEvaluateContext context, Trilu target)
    {
        return context.GetArgumentShape(target, Trilu.Input);
    }
}
