// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase;

/// <summary>
/// Unary operator kind.
/// </summary>
public enum UnaryOp : byte
{
    /// <summary>
    /// Abs.
    /// </summary>
    Abs,

    /// <summary>
    /// Acos.
    /// </summary>
    Acos,

    /// <summary>
    /// Acosh.
    /// </summary>
    Acosh,

    /// <summary>
    /// Asin.
    /// </summary>
    Asin,

    /// <summary>
    /// Asin.
    /// </summary>
    Asinh,

    /// <summary>
    /// Ceil.
    /// </summary>
    Ceil,

    /// <summary>
    /// Cosine.
    /// </summary>
    Cos,

    /// <summary>
    /// Cosh.
    /// </summary>
    Cosh,

    /// <summary>
    /// Erf.
    /// </summary>
    Erf,

    /// <summary>
    /// Exp.
    /// </summary>
    Exp,

    /// <summary>
    /// Floor.
    /// </summary>
    Floor,

    /// <summary>
    /// Log.
    /// </summary>
    Log,

    /// <summary>
    /// Neg.
    /// </summary>
    Neg,

    /// <summary>
    /// Round.
    /// </summary>
    Round,

    /// <summary>
    /// Rsqrt.
    /// </summary>
    Rsqrt,

    /// <summary>
    /// Sine.
    /// </summary>
    Sin,

    /// <summary>
    /// Sinh.
    /// </summary>
    Sinh,

    /// <summary>
    /// Sign.
    /// </summary>
    Sign,

    /// <summary>
    /// Sqrt.
    /// </summary>
    Sqrt,

    /// <summary>
    /// Square.
    /// </summary>
    Square,

    /// <summary>
    /// Tanh.
    /// </summary>
    Tanh,

    /// <summary>
    /// Bitwise not.
    /// </summary>
    BitwiseNot,

    /// <summary>
    /// Logical not.
    /// </summary>
    LogicalNot,
}
