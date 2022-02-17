// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.IR;
using Nncase.IR.Math;

namespace Nncase.Pattern;

/// <summary>
/// Pattern for <see cref="Expr"/>.
/// </summary>
/// <param name="Condition">Expression condition.</param>
public sealed record ExprPattern(Func<Expr, bool> Condition)
    : Pattern<Expr>(Condition)
{
}

public static partial class Utility
{
    /// <summary>
    /// Get the current expr checked Shape.
    /// </summary>
    /// <param name="expr"></param>
    /// <returns></returns>
    /// <exception cref="InvalidOperationException"></exception>
    public static List<Dimension> GetShape(Expr expr) => expr.CheckedType switch
    {
        TensorType type => new List<Dimension>(type.Shape),
        _ => throw new InvalidOperationException($"The Expr {expr.GetType().Name} Has No Shape!"),
    };
}
