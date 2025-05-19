// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Buffers;

namespace Nncase.Evaluator.Buffers;

/// <summary>
/// Evaluator for AddressOf.
/// </summary>
public partial class BufferSubviewEvaluator : ITypeInferencer<BufferSubview>
{
    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, BufferSubview target)
    {
        var buffer = context.GetArgument(target, BufferSubview.Buffer);
        var shape = (Shape)context.GetArgument(target, BufferSubview.Shape);
        return new TensorType(buffer.CheckedDataType, shape);
    }
}
