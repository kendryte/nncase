// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.Tensors;

namespace Nncase.Evaluator.Tensors;

/// <summary>
/// Evaluator for <see cref="LocalShardDim"/>.
/// </summary>
public class LocalShardDimEvaluator : ITypeInferencer<LocalShardDim>
{
    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, LocalShardDim target)
    {
        var dimtype = context.CheckArgumentType<DimensionType>(target, LocalShardDim.Dim);
        return dimtype;
    }
}
