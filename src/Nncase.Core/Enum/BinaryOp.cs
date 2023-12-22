// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase;

/// <summary>
/// Binary opeartor kind.
/// </summary>
public enum BinaryOp : byte
{
    /// <summary>
    /// Add.
    /// </summary>
    Add,

    /// <summary>
    /// Sub.
    /// </summary>
    Sub,

    /// <summary>
    /// Multiply.
    /// </summary>
    Mul,

    /// <summary>
    /// Divide.
    /// </summary>
    Div,

    /// <summary>
    /// Modulus.
    /// </summary>
    Mod,

    /// <summary>
    /// Minimum.
    /// </summary>
    Min,

    /// <summary>
    /// Maximum.
    /// </summary>
    Max,

    /// <summary>
    /// Power.
    /// </summary>
    Pow,

    /// <summary>
    /// Bitwise and.
    /// </summary>
    BitwiseAnd,

    /// <summary>
    /// Bitwise or.
    /// </summary>
    BitwiseOr,

    /// <summary>
    /// Bitwise xor.
    /// </summary>
    BitwiseXor,

    /// <summary>
    /// Logical and.
    /// </summary>
    LogicalAnd,

    /// <summary>
    /// Logical or.
    /// </summary>
    LogicalOr,

    /// <summary>
    /// Logical xor.
    /// </summary>
    LogicalXor,

    /// <summary>
    /// Left Shift.
    /// </summary>
    LeftShift,

    /// <summary>
    /// Right Shift.
    /// </summary>
    RightShift,

    /// <summary>
    /// Floor Div.
    /// </summary>
    FloorDiv,

    /// <summary>
    /// Ceil Div.
    /// </summary>
    CeilDiv,
}
