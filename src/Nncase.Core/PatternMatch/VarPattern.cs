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
public sealed record VarPattern : Pattern<Var>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="VarPattern"/> class.
    /// </summary>
    /// <param name="typePattern">Type pattern.</param>
    public VarPattern(TypePattern typePattern)
    {
        TypePattern = typePattern;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="VarPattern"/> class.
    /// </summary>
    /// <param name="var">Var expression.</param>
    public VarPattern(Var var)
        : this(new TypePattern(var.TypeAnnotation))
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="VarPattern"/> class.
    /// </summary>
    public VarPattern()
        : this(new TypePattern(AnyType.Default))
    {
    }
}

public static partial class Utility
{
    public static VarPattern IsVar(TypePattern Type) => new VarPattern(Type);

    public static VarPattern IsVar() => new VarPattern(IsAnyType());
}
