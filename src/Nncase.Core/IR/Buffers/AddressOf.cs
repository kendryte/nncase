﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR.Tensors;
using Nncase.PatternMatch;
using static Nncase.IR.TypePatternUtility;

namespace Nncase.IR.Buffers;

/// <summary>
/// AddressOf expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class AddressOf : Op
{
    /// <summary>
    /// Get the input parameter.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(AddressOf), 0, "input");

    /// <inheritdoc/>
    public override bool CanFoldConstCall => false;
}
