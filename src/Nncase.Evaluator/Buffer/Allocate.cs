// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Autofac;
using Nncase.IR;
using Nncase.IR.Buffer;

namespace Nncase.Evaluator.Buffer;

/// <summary>
/// Evaluator for DDrOf 
/// </summary>
public partial class AllocateEvaluator : ITypeInferencer<Allocate>
{
    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Allocate target)
    {
        return TensorType.Pointer(target.ElemType.DType);
    }
}