// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.IR;
using Nncase.IR.Shapes;
using Nncase.IR.Tensors;

namespace Nncase.PatternMatch;

/// <summary>
/// Pattern for <see cref="Padding"/>.
/// </summary>
/// <param name="Before">Before pattern.</param>
/// <param name="After">After pattern.</param>
/// <param name="Name"> name. </param>
public sealed record PaddingPattern(Pattern Before, Pattern After, string? Name) : Pattern<Padding>(Name)
{
    /// <summary>
    /// Initializes a new instance of the <see cref="PaddingPattern"/> class.
    /// </summary>
    /// <param name="padding"><see cref="Padding"/> expression.</param>
    /// <param name="name">name.</param>
    public PaddingPattern(Padding padding, string? name)
        : this(padding.Before, padding.After, name)
    {
    }
}

/// <summary>
/// PatternMatch Utility.
/// </summary>
public static partial class Utility
{
    /// <summary>
    /// is call .
    /// </summary>
    /// <param name="name">name.</param>
    /// <param name="before">before.</param>
    /// <param name="after">after.</param>
    /// <returns>call pattern.</returns>
    public static PaddingPattern IsPadding(string? name = null, Pattern? before = null, Pattern? after = null) => new PaddingPattern(before ?? IsWildcard(), after ?? IsWildcard(), name);

    public static PaddingPattern IsFixedPadding(string? name = null) => new PaddingPattern(IsFixedDimension(), IsFixedDimension(), name);
}
