// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.IR;
using Nncase.IR.Shapes;
using Nncase.IR.Tensors;

namespace Nncase.PatternMatch;

/// <summary>
/// Pattern for <see cref="Paddings"/>.
/// </summary>
/// <param name="Values">Arguments pattern.</param>
/// <param name="Name"> name. </param>
public sealed record PaddingsPattern(VArgsPattern Values, string? Name) : Pattern<Paddings>(Name)
{
    /// <summary>
    /// Initializes a new instance of the <see cref="PaddingsPattern"/> class.
    /// </summary>
    /// <param name="paddings"><see cref="Paddings"/> expression.</param>
    /// <param name="name">name.</param>
    public PaddingsPattern(Paddings paddings, string? name)
        : this(new VArgsPattern(paddings.Values.ToArray(), null), name)
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
    /// <param name="values">values.</param>
    /// <returns>call pattern.</returns>
    public static PaddingsPattern IsPaddings(string? name = null, VArgsPattern? values = null) => new PaddingsPattern(values ?? IsVArgsRepeat(IsWildcard), name);

    public static PaddingsPattern IsFixedPaddings(string? name = null, VArgsPattern? values = null) => new PaddingsPattern(values ?? IsVArgsRepeat(() => IsFixedPadding()), name);
}
