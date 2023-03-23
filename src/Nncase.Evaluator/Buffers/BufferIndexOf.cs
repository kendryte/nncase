// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.Buffers;

namespace Nncase.Evaluator.Buffers;

/// <summary>
/// Evaluator for BufferIndexOf.
/// </summary>
[TypeInferGenerator]
public partial class BufferIndexOfEvaluator : ITypeInferencer<BufferIndexOf>
{
    private IRType Visit(TensorType input)
    {
        return new TensorType(DataTypes.Int32, Shape.Scalar);
    }
}
