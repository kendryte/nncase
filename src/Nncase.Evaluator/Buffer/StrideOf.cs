// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.IO;
using System.Runtime.Serialization.Formatters.Binary;
using Nncase.IR;
using Nncase.IR.Buffer;

namespace Nncase.Evaluator.Buffer;

/// <summary>
/// Evaluator for <see cref="StrideOf"/>.
/// </summary>
[TypeInferGenerator]
public partial class StrideOfEvaluator : ITypeInferencer<StrideOf>
{
    IRType Visit(TensorType Input) => new TensorType(DataTypes.Int32, new[] { Input.Shape.Count });
}
