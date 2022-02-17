// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static Nncase.Pattern.F.Math;

namespace Nncase.Pattern;

/// <summary>
/// Math operators for <see cref="ExprPattern"/>.
/// </summary>
public partial record Pattern
{
    /// <summary>
    /// Unary neg.
    /// </summary>
    /// <param name="lhs">Source operand.</param>
    /// <returns>Result.</returns>
    public static CallPattern operator -(Pattern lhs) => Neg(lhs);

    /// <summary>
    /// Unary bitwise not.
    /// </summary>
    /// <param name="lhs">Source operand.</param>
    /// <returns>Result.</returns>
    public static CallPattern operator ~(Pattern lhs) => BitwiseNot(lhs);

    /// <summary>
    /// Unary logical not.
    /// </summary>
    /// <param name="lhs">Source operand.</param>
    /// <returns>Result.</returns>
    public static CallPattern operator !(Pattern lhs) => LogicalNot(lhs);

    /// <summary>
    /// Binary add.
    /// </summary>
    /// <param name="lhs">Left operand.</param>
    /// <param name="rhs">Right operand.</param>
    /// <returns>Result.</returns>
    public static CallPattern operator +(Pattern lhs, Pattern rhs) => Add(lhs, rhs);

    /// <summary>
    /// Binary sub.
    /// </summary>
    /// <param name="lhs">Left operand.</param>
    /// <param name="rhs">Right operand.</param>
    /// <returns>Result.</returns>
    public static CallPattern operator -(Pattern lhs, Pattern rhs) => Sub(lhs, rhs);

    /// <summary>
    /// Binary mul.
    /// </summary>
    /// <param name="lhs">Left operand.</param>
    /// <param name="rhs">Right operand.</param>
    /// <returns>Result.</returns>
    public static CallPattern operator *(Pattern lhs, Pattern rhs) => Mul(lhs, rhs);

    /// <summary>
    /// Binary div.
    /// </summary>
    /// <param name="lhs">Left operand.</param>
    /// <param name="rhs">Right operand.</param>
    /// <returns>Result.</returns>
    public static CallPattern operator /(Pattern lhs, Pattern rhs) => Div(lhs, rhs);

    /// <summary>
    /// Binary mod.
    /// </summary>
    /// <param name="lhs">Left operand.</param>
    /// <param name="rhs">Right operand.</param>
    /// <returns>Result.</returns>
    public static CallPattern operator %(Pattern lhs, Pattern rhs) => Mod(lhs, rhs);

    /// <summary>
    /// Binary bitwise and.
    /// </summary>
    /// <param name="lhs">Left operand.</param>
    /// <param name="rhs">Right operand.</param>
    /// <returns>Result.</returns>
    public static CallPattern operator &(Pattern lhs, Pattern rhs) => BitwiseAnd(lhs, rhs);

    /// <summary>
    /// Binary bitwise or.
    /// </summary>
    /// <param name="lhs">Left operand.</param>
    /// <param name="rhs">Right operand.</param>
    /// <returns>Result.</returns>
    public static CallPattern operator |(Pattern lhs, Pattern rhs) => BitwiseOr(lhs, rhs);

    /// <summary>
    /// Binary bitwise xor.
    /// </summary>
    /// <param name="lhs">Left operand.</param>
    /// <param name="rhs">Right operand.</param>
    /// <returns>Result.</returns>
    public static CallPattern operator ^(Pattern lhs, Pattern rhs) => BitwiseXor(lhs, rhs);
}
