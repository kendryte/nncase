﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.PatternMatch;

namespace Nncase.IR.Tensors;

/// <summary>
/// Devectorize expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class Unpack : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(Unpack), 0, "input", ParameterKind.Input);

    public IRArray<int> Lanes { get; }

    public IRArray<int> Axes { get; }

    /// <inheritdoc/>
    public override string DisplayProperty() => $"Lanes: {Lanes}, Axes: {Axes}";
}
