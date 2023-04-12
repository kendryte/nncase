// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.K210;
using OrtKISharp;

namespace Nncase.Evaluator.K210;

/// <summary>
/// Evaluator for <see cref="FakeKPUDownload"/>.
/// </summary>
internal sealed class FakeKPUDownloadEvaluator : IEvaluator<FakeKPUDownload>, ITypeInferencer<FakeKPUDownload>, ICostEvaluator<FakeKPUDownload>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, FakeKPUDownload target)
    {
        var input = context.GetArgumentValue(target, FakeKPUDownload.Input);
        return input;
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, FakeKPUDownload target)
    {
        var input = context.CheckArgumentType<TensorType>(target, FakeKPUDownload.Input);
        return Visit(target, input);
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, FakeKPUDownload target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, FakeKPUDownload.Input);
        var outputType = context.GetReturnType<TensorType>();

        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetFakeMemoryAccess(inputType, 8),
            [CostFactorNames.MemoryStore] = CostUtility.GetFakeMemoryAccess(outputType, 8),
        };
    }

    private IRType Visit(FakeKPUDownload target, TensorType input)
    {
        return input;
    }
}
