// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.K210;
using OrtKISharp;

namespace Nncase.Evaluator.K210;

/// <summary>
/// Evaluator for <see cref="KPUDownload"/>.
/// </summary>
internal sealed class KPUDownloadEvaluator : IEvaluator<KPUDownload>, ITypeInferencer<KPUDownload>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, KPUDownload target)
    {
        var input = context.GetArgumentValue(target, KPUDownload.Input);
        return input;
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, KPUDownload target)
    {
        var input = context.CheckArgumentType<TensorType>(target, KPUDownload.Input);
        return Visit(target, input);
    }

    private IRType Visit(KPUDownload target, TensorType input)
    {
        return input;
    }
}
