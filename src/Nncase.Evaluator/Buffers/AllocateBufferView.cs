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
        return buffer.CheckedType;
    }
}
