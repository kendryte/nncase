// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.Buffers;

namespace Nncase.Evaluator.Buffers;

/// <summary>
/// Evaluator for DDrOf.
/// </summary>
public partial class AllocateBufferViewEvaluator : ITypeInferencer<AllocateBufferView>
{
    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, AllocateBufferView target)
    {
        var buffer = (Nncase.TIR.Buffer)context.GetArgument(target, AllocateBufferView.Buffer);

        // TODO: fixed shape
        return new TensorType(buffer.ElemType, Shape.Unknown(buffer.Rank));
    }
}
