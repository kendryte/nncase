// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Linq;
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
        return new TensorType(buffer.ElemType, new Shape(buffer.Dimensions.ToArray().Select(d => d is TensorConst tc ? new Dimension(tc.Value.ToScalar<int>()) : Dimension.Unknown)));
    }
}
