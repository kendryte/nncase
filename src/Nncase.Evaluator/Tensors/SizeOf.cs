// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Tensors;
using OrtKISharp;

namespace Nncase.Evaluator.Tensors;

/// <summary>
/// Evaluator for <see cref="SizeOf"/>.
/// </summary>
public class SizeOfEvaluator : IEvaluator<SizeOf>, ITypeInferencer<SizeOf>, ICostEvaluator<SizeOf>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, SizeOf size)
    {
        var input = context.GetOrtArgumentValue(size, SizeOf.Input);
        return OrtKI.Size(input).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, SizeOf target)
    {
        return new TensorType(DataTypes.Int64, Shape.Scalar);
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, SizeOf target)
    {
        return new()
        {
            [CostFactorNames.CPUCycles] = 1,
        };
    }
}
