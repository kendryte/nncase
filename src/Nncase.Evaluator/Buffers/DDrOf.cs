﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.Buffers;

namespace Nncase.Evaluator.Buffers;

/// <summary>
/// Evaluator for AddressOf.
/// </summary>
[TypeInferGenerator]
public partial class AddressOfEvaluator : ITypeInferencer<AddressOf>
{
    private IRType Visit(IRType input)
    {
        return input switch
        {
            DistributedType d => TensorType.Pointer(d.TensorType.DType),
            TensorType t => TensorType.Pointer(t.DType),
            _ => new InvalidType(input.GetType().Name),
        };
    }
}
