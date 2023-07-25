// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR.Tensors;
using Nncase.PatternMatch;
using static Nncase.IR.TypePatternUtility;

namespace Nncase.IR.Buffers;

/// <summary>
/// BufferIndexOf expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class BufferLoad : Op
{
    /// <summary>
    /// Get the input parameter.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(BufferLoad), 0, "input", IsTensor());
    
    public static readonly ParameterInfo Indices = new(typeof(BufferLoad), 1, "indices", IsTuple());
}
