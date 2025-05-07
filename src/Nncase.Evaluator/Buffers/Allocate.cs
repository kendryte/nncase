// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.Buffers;

namespace Nncase.Evaluator.Buffers;

/// <summary>
/// Evaluator for AddressOf.
/// </summary>
public partial class AllocateEvaluator : ITypeInferencer<Allocate>
{
    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Allocate target)
    {
        return TensorType.Pointer(target.ElemType);
    }
}
