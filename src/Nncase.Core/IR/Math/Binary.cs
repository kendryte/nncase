// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.Pattern;

namespace Nncase.IR.Math;

/// <summary>
/// Binary expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed record Binary(BinaryOp BinaryOp) : Op
{
    /// <summary>
    /// Gets lhs.
    /// </summary>
    public static readonly ParameterInfo Lhs = new(typeof(Binary), 0, "lhs");

    /// <summary>
    /// Gets rhs.
    /// </summary>
    public static readonly ParameterInfo Rhs = new(typeof(Binary), 1, "rhs");

    /// <summary>
    /// convert Binary Op to literal.
    /// <example>
    ///   BinaryOp.Add => "+"
    /// </example>
    /// </summary>
    /// <returns></returns>
    public string ToLiteral() => BinaryOp switch
    {
        BinaryOp.Add => "+",
        BinaryOp.Sub => "-",
        BinaryOp.Mul => "*",
        BinaryOp.Div => "/",
        _ => throw new NotSupportedException($"{BinaryOp}"),
    };
}
