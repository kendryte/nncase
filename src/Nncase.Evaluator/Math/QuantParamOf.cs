// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.Quantization;
using Nncase.Utilities;
using OrtKISharp;

namespace Nncase.Evaluator.Math;

/// <summary>
/// Evaluator for <see cref="QuantParamOf"/>.
/// </summary>
public class QuantParamOfEvaluator : IEvaluator<QuantParamOf>, ITypeInferencer<QuantParamOf>, ICostEvaluator<QuantParamOf>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, QuantParamOf target)
    {
        var rawRange = context.GetArgumentValueAsArray<float>(target, QuantParamOf.Range);
        var bits = context.GetArgumentValueAsScalar<int>(target, QuantParamOf.Bits);
        var min = rawRange[0];
        var max = rawRange[1];
        return Value.FromTensor(Tensor.FromScalar(QuantUtility.GetQuantParam((min, max), bits, target.QuantMode)));
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, QuantParamOf target)
    {
        _ = context.CheckArgumentType<TensorType>(target, QuantParamOf.Range);
        return new TensorType(new QuantParamType(), Shape.Scalar);
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, QuantParamOf target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, QuantParamOf.Range);
        var outputType = context.GetReturnType<TensorType>();

        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(inputType),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(outputType),
            [CostFactorNames.CPUCycles] = 2,
        };
    }
}
