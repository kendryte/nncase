// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.Buffers;

namespace Nncase.Evaluator.Buffers;

/// <summary>
/// Evaluator for DDrOf.
/// </summary>
[TypeInferGenerator]
public partial class BaseMentOfEvaluator : ITypeInferencer<BaseMentOf>
{
    private IRType Visit(TensorType input)
    {
        return TensorType.Scalar(DataTypes.Int32);
    }
}
