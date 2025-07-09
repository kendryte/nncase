// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.PatternMatch;

namespace Nncase.IR.Tensors;

/// <summary>
/// PackMask expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class PackMask : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(PackMask), 0, "input", ParameterKind.Input);

    public MaskVectorStyle Style { get; }

    public int ElementBits { get; }

    public int Lanes { get; }

    public int Axis { get; }

    /// <inheritdoc/>
    public override string DisplayProperty() => $"Style: {Style}, ElementBits: {ElementBits}, Lanes: {Lanes}, Axis: {Axis}";
}
