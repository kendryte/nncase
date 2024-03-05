// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using NetFabric.Hyperlinq;
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
        var buffer = context.GetArgument(target, AllocateBufferView.Buffer);
        var shapeExpr = context.GetArgument(target, AllocateBufferView.Shape);
        var shape = shapeExpr switch
        {
            IR.Tuple t => new Shape(t.Fields.AsValueEnumerable().Select(d => d is TensorConst tc ? new Dimension(tc.Value.ToScalar<int>()) : Dimension.Unknown).ToArray()),
            TupleConst tc => new Shape(tc.Value.Select(d => d is Tensor t ? new Dimension(t.ToScalar<int>()) : Dimension.Unknown)),
            _ => throw new ArgumentException("Invalid shape argument."),
        };
        return new TensorType(buffer.CheckedDataType, shape);
    }
}
