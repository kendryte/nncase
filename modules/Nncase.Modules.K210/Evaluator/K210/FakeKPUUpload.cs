// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.K210;
using OrtKISharp;

namespace Nncase.Evaluator.K210;

/// <summary>
/// Evaluator for <see cref="FakeKPUUpload"/>.
/// </summary>
internal sealed class FakeKPUUploadEvaluator : IEvaluator<FakeKPUUpload>, ITypeInferencer<FakeKPUUpload>, ICostEvaluator<FakeKPUUpload>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, FakeKPUUpload target)
    {
        var input = context.GetArgumentValue(target, FakeKPUUpload.Input);
        return input;
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, FakeKPUUpload target)
    {
        var input = context.CheckArgumentType<TensorType>(target, FakeKPUUpload.Input);
        return Visit(target, input);
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, FakeKPUUpload target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, FakeKPUUpload.Input);
        var outputType = context.GetReturnType<TensorType>();

        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetFakeMemoryAccess(inputType, 8),
            [CostFactorNames.MemoryStore] = CostUtility.GetFakeMemoryAccess(outputType, 8),
        };
    }

    private IRType Visit(FakeKPUUpload target, TensorType input)
    {
        return input;
    }
}
