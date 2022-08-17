
// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR.Tensors;
using Nncase.PatternMatch;
using static Nncase.IR.TypePatternUtility;

namespace Nncase.IR.Buffer;

/// <summary>
/// get the buffer from the input.
/// </summary>
public record BufferOf(Schedule.MemoryLocation MemoryLocation) : Op
{
    /// <summary>
    /// Get the input parameter.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(BufferOf), 0, "input", IsTensor());
}