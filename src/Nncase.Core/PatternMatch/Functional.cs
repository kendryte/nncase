// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.PatternMatch.F;

/// <summary>
/// Math functional helper.
/// </summary>
public static partial class Math
{
    /// <summary>
    /// CallPattern abs.
    /// </summary>
    /// <param name="expr">Source expression.</param>
    /// <returns>Result expression.</returns>
    public static CallPattern Abs(Pattern expr) => IsUnary(UnaryOp.Abs, expr);

    /// <summary>
    /// CallPattern ceil.
    /// </summary>
    /// <param name="expr">Source expression.</param>
    /// <returns>Result expression.</returns>
    public static CallPattern Ceil(Pattern expr) => IsUnary(UnaryOp.Ceil, expr);

    /// <summary>
    /// CallPattern cos.
    /// </summary>
    /// <param name="expr">Source expression.</param>
    /// <returns>Result expression.</returns>
    public static CallPattern Cos(Pattern expr) => IsUnary(UnaryOp.Cos, expr);

    /// <summary>
    /// CallPattern exp.
    /// </summary>
    /// <param name="expr">Source expression.</param>
    /// <returns>Result expression.</returns>
    public static CallPattern Exp(Pattern expr) => IsUnary(UnaryOp.Exp, expr);

    /// <summary>
    /// CallPattern floor.
    /// </summary>
    /// <param name="expr">Source expression.</param>
    /// <returns>Result expression.</returns>
    public static CallPattern Floor(Pattern expr) => IsUnary(UnaryOp.Floor, expr);

    /// <summary>
    /// CallPattern log.
    /// </summary>
    /// <param name="expr">Source expression.</param>
    /// <returns>Result expression.</returns>
    public static CallPattern Log(Pattern expr) => IsUnary(UnaryOp.Log, expr);

    /// <summary>
    /// CallPattern neg.
    /// </summary>
    /// <param name="expr">Source expression.</param>
    /// <returns>Result expression.</returns>
    public static CallPattern Neg(Pattern expr) => IsUnary(UnaryOp.Neg, expr);

    /// <summary>
    /// CallPattern round.
    /// </summary>
    /// <param name="expr">Source expression.</param>
    /// <returns>Result expression.</returns>
    public static CallPattern Round(Pattern expr) => IsUnary(UnaryOp.Round, expr);

    /// <summary>
    /// CallPattern rsqrt.
    /// </summary>
    /// <param name="expr">Source expression.</param>
    /// <returns>Result expression.</returns>
    public static CallPattern Rsqrt(Pattern expr) => IsUnary(UnaryOp.Rsqrt, expr);

    /// <summary>
    /// CallPattern sin.
    /// </summary>
    /// <param name="expr">Source expression.</param>
    /// <returns>Result expression.</returns>
    public static CallPattern Sin(Pattern expr) => IsUnary(UnaryOp.Sin, expr);

    /// <summary>
    /// CallPattern sqrt.
    /// </summary>
    /// <param name="expr">Source expression.</param>
    /// <returns>Result expression.</returns>
    public static CallPattern Sqrt(Pattern expr) => IsUnary(UnaryOp.Sqrt, expr);

    /// <summary>
    /// CallPattern square.
    /// </summary>
    /// <param name="expr">Source expression.</param>
    /// <returns>Result expression.</returns>
    public static CallPattern Square(Pattern expr) => IsUnary(UnaryOp.Square, expr);

    /// <summary>
    /// CallPattern tanh.
    /// </summary>
    /// <param name="expr">Source expression.</param>
    /// <returns>Result expression.</returns>
    public static CallPattern Tanh(Pattern expr) => IsUnary(UnaryOp.Tanh, expr);

    /// <summary>
    /// CallPattern bitwise not.
    /// </summary>
    /// <param name="expr">Source expression.</param>
    /// <returns>Result expression.</returns>
    public static CallPattern BitwiseNot(Pattern expr) => IsUnary(UnaryOp.BitwiseNot, expr);

    /// <summary>
    /// CallPattern logical not.
    /// </summary>
    /// <param name="expr">Source expression.</param>
    /// <returns>Result expression.</returns>
    public static CallPattern LogicalNot(Pattern expr) => IsUnary(UnaryOp.LogicalNot, expr);

    /// <summary>
    /// CallPattern add.
    /// </summary>
    /// <param name="lhs">Left operand.</param>
    /// <param name="rhs">Right operand.</param>
    /// <returns>Result expression.</returns>
    public static CallPattern Add(Pattern lhs, Pattern rhs) => IsBinary(BinaryOp.Add, lhs, rhs);

    /// <summary>
    /// CallPattern sub.
    /// </summary>
    /// <param name="lhs">Left operand.</param>
    /// <param name="rhs">Right operand.</param>
    /// <returns>Result expression.</returns>
    public static CallPattern Sub(Pattern lhs, Pattern rhs) => IsBinary(BinaryOp.Sub, lhs, rhs);

    /// <summary>
    /// CallPattern mul.
    /// </summary>
    /// <param name="lhs">Left operand.</param>
    /// <param name="rhs">Right operand.</param>
    /// <returns>Result expression.</returns>
    public static CallPattern Mul(Pattern lhs, Pattern rhs) => IsBinary(BinaryOp.Mul, lhs, rhs);

    /// <summary>
    /// CallPattern div.
    /// </summary>
    /// <param name="lhs">Left operand.</param>
    /// <param name="rhs">Right operand.</param>
    /// <returns>Result expression.</returns>
    public static CallPattern Div(Pattern lhs, Pattern rhs) => IsBinary(BinaryOp.Div, lhs, rhs);

    /// <summary>
    /// CallPattern mod.
    /// </summary>
    /// <param name="lhs">Left operand.</param>
    /// <param name="rhs">Right operand.</param>
    /// <returns>Result expression.</returns>
    public static CallPattern Mod(Pattern lhs, Pattern rhs) => IsBinary(BinaryOp.Mod, lhs, rhs);

    /// <summary>
    /// CallPattern min.
    /// </summary>
    /// <param name="lhs">Left operand.</param>
    /// <param name="rhs">Right operand.</param>
    /// <returns>Result expression.</returns>
    public static CallPattern Min(Pattern lhs, Pattern rhs) => IsBinary(BinaryOp.Min, lhs, rhs);

    /// <summary>
    /// CallPattern max.
    /// </summary>
    /// <param name="lhs">Left operand.</param>
    /// <param name="rhs">Right operand.</param>
    /// <returns>Result expression.</returns>
    public static CallPattern Max(Pattern lhs, Pattern rhs) => IsBinary(BinaryOp.Max, lhs, rhs);

    /// <summary>
    /// CallPattern pow.
    /// </summary>
    /// <param name="lhs">Left operand.</param>
    /// <param name="rhs">Right operand.</param>
    /// <returns>Result expression.</returns>
    public static CallPattern Pow(Pattern lhs, Pattern rhs) => IsBinary(BinaryOp.Pow, lhs, rhs);

    /// <summary>
    /// CallPattern bitwise and.
    /// </summary>
    /// <param name="lhs">Left operand.</param>
    /// <param name="rhs">Right operand.</param>
    /// <returns>Result expression.</returns>
    public static CallPattern BitwiseAnd(Pattern lhs, Pattern rhs) => IsBinary(BinaryOp.BitwiseAnd, lhs, rhs);

    /// <summary>
    /// CallPattern bitwise or.
    /// </summary>
    /// <param name="lhs">Left operand.</param>
    /// <param name="rhs">Right operand.</param>
    /// <returns>Result expression.</returns>
    public static CallPattern BitwiseOr(Pattern lhs, Pattern rhs) => IsBinary(BinaryOp.BitwiseOr, lhs, rhs);

    /// <summary>
    /// CallPattern bitwise xor.
    /// </summary>
    /// <param name="lhs">Left operand.</param>
    /// <param name="rhs">Right operand.</param>
    /// <returns>Result expression.</returns>
    public static CallPattern BitwiseXor(Pattern lhs, Pattern rhs) => IsBinary(BinaryOp.BitwiseXor, lhs, rhs);

    /// <summary>
    /// CallPattern logical and.
    /// </summary>
    /// <param name="lhs">Left operand.</param>
    /// <param name="rhs">Right operand.</param>
    /// <returns>Result expression.</returns>
    public static CallPattern LogicalAnd(Pattern lhs, Pattern rhs) => IsBinary(BinaryOp.LogicalAnd, lhs, rhs);

    /// <summary>
    /// CallPattern logical or.
    /// </summary>
    /// <param name="lhs">Left operand.</param>
    /// <param name="rhs">Right operand.</param>
    /// <returns>Result expression.</returns>
    public static CallPattern LogicalOr(Pattern lhs, Pattern rhs) => IsBinary(BinaryOp.LogicalOr, lhs, rhs);

    /// <summary>
    /// CallPattern logical xor.
    /// </summary>
    /// <param name="lhs">Left operand.</param>
    /// <param name="rhs">Right operand.</param>
    /// <returns>Result expression.</returns>
    public static CallPattern LogicalXor(Pattern lhs, Pattern rhs) => IsBinary(BinaryOp.LogicalXor, lhs, rhs);

    /// <summary>
    /// CallPattern left shift.
    /// </summary>
    /// <param name="lhs">Left operand.</param>
    /// <param name="rhs">Right operand.</param>
    /// <returns>Result expression.</returns>
    public static CallPattern LeftShift(Pattern lhs, Pattern rhs) => IsBinary(BinaryOp.LeftShift, lhs, rhs);

    /// <summary>
    /// CallPattern right shift.
    /// </summary>
    /// <param name="lhs">Left operand.</param>
    /// <param name="rhs">Right operand.</param>
    /// <returns>Result expression.</returns>
    public static CallPattern RightShift(Pattern lhs, Pattern rhs) => IsBinary(BinaryOp.RightShift, lhs, rhs);

    /// <summary>
    /// CallPattern floor div.
    /// </summary>
    /// <param name="lhs">Left operand.</param>
    /// <param name="rhs">Right operand.</param>
    /// <returns>Result expression.</returns>
    public static CallPattern FloorDiv(Pattern lhs, Pattern rhs) => Floor(lhs / rhs);

    /// <summary>
    /// CallPattern floor mod.
    /// </summary>
    /// <param name="lhs">Left operand.</param>
    /// <param name="rhs">Right operand.</param>
    /// <returns>Result expression.</returns>
    public static CallPattern FloorMod(Pattern lhs, Pattern rhs) => Sub(lhs, FloorDiv(lhs, rhs) * rhs);

    /// <summary>
    /// CallPattern equal.
    /// </summary>
    /// <returns>result.</returns>
    public static CallPattern Equal(Pattern lhs, Pattern rhs) => IsCompare(CompareOp.Equal, lhs, rhs);

    /// <summary>
    /// CallPattern not equal.
    /// </summary>
    public static CallPattern NotEqual(Pattern lhs, Pattern rhs) => IsCompare(CompareOp.NotEqual, lhs, rhs);

    /// <summary>
    /// CallPattern less than.
    /// </summary>
    public static CallPattern LessThan(Pattern lhs, Pattern rhs) => IsCompare(CompareOp.LowerThan, lhs, rhs);

    /// <summary>
    /// CallPattern less equal.
    /// </summary>
    public static CallPattern LessEqual(Pattern lhs, Pattern rhs) => IsCompare(CompareOp.LowerOrEqual, lhs, rhs);

    /// <summary>
    /// CallPattern greater equal.
    /// </summary>
    public static CallPattern GreaterEqual(Pattern lhs, Pattern rhs) => IsCompare(CompareOp.GreaterThan, lhs, rhs);

    /// <summary>
    /// CallPattern greater than.
    /// </summary>
    public static CallPattern GreaterThan(Pattern lhs, Pattern rhs) => IsCompare(CompareOp.GreaterOrEqual, lhs, rhs);
}
