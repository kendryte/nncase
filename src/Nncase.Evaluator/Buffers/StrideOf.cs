// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.IO;
using System.Runtime.Serialization.Formatters.Binary;
using Nncase.IR;
using Nncase.IR.Buffers;

namespace Nncase.Evaluator.Buffers;

/// <summary>
/// Evaluator for <see cref="StrideOf"/>.
/// </summary>
[TypeInferGenerator]
public partial class StrideOfEvaluator : ITypeInferencer<StrideOf>
{
    private IRType Visit(TensorType input) => new TensorType(DataTypes.Int32, new[] { input.Shape.Count });
}
