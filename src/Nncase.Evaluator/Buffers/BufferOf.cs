// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.Buffers;

namespace Nncase.Evaluator.Buffers;

/// <summary>
/// Evaluator for BufferOf.
/// </summary>
[TypeInferGenerator]
public partial class BufferOfEvaluator : ITypeInferencer<BufferOf>
{
    private IRType Visit(TensorType input)
    {
        return TupleType.Void;
    }
}
