﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR.Tensors;
using Nncase.PatternMatch;
using static Nncase.IR.TypePatternUtility;

namespace Nncase.IR.Buffers;

/// <summary>
/// BufferStore op.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class BufferStore : Op
{
    /// <summary>
    /// Get the input parameter.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(BufferStore), 0, "input", IsTensor());

    /// <summary>
    /// Get the indices parameter.
    /// </summary>
    public static readonly ParameterInfo Indices = new(typeof(BufferStore), 1, "indices", IsTuple());

    /// <summary>
    /// Get the value parameter.
    /// </summary>
    public static readonly ParameterInfo Value = new(typeof(BufferStore), 2, "value", IsScalar());

    /// <inheritdoc/>
    public override bool CanFoldConstCall => false;
}
