// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Nncase.IR;
using static Nncase.IR.TypePatternUtility;

namespace Nncase.PatternMatch;

/// <summary>
/// Pattern for <see cref="Var"/>.
/// </summary>
/// <param name="Name">name.</param>
public sealed record VarPattern(string? Name) : Pattern<Var>(Name)
{
    /// <summary>
    /// Initializes a new instance of the <see cref="VarPattern"/> class.
    /// </summary>
    /// <param name="var">Var expression.</param>
    public VarPattern(Var var)
        : this(var.Name)
    {
        TypePattern = new TypePattern(var.TypeAnnotation);
    }
}

public static partial class Utility
{
    /// <summary>
    /// create var pattern.
    /// </summary>
    /// <param name="typePattern">type pattern.</param>
    /// <param name="name">name.</param>
    /// <returns>the var pattern.</returns>
    public static VarPattern IsVar(TypePattern typePattern, string? name = null) => new VarPattern(name) with { TypePattern = typePattern };

    /// <summary>
    /// create var pattern.
    /// </summary>
    /// <param name="name">name.</param>
    /// <returns>the var pattern.</returns>
    public static VarPattern IsVar(string? name = null) => new VarPattern(name) with { TypePattern = IsIRType() };
}
