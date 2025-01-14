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
public partial class BufferSubviewEvaluator : ITypeInferencer<BufferSubview>
{
    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, BufferSubview target)
    {
        var buffer = context.GetArgument(target, BufferSubview.Buffer);
        var shapeExpr = context.GetArgument(target, BufferSubview.Shape);
        var shape = shapeExpr switch
        {
            IR.Tuple t => new Shape(t.Fields),
            TupleConst tc => new Shape(tc.Value.AsTensor().ToArray<long>()),
            _ => throw new ArgumentException("Invalid shape argument."),
        };
        return new TensorType(buffer.CheckedDataType, shape);
    }
}
