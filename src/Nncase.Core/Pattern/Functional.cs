// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.Pattern.F
{
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
        public static CallPattern Abs(Pattern expr) => Unary(expr, UnaryOp.Abs);

        /// <summary>
        /// CallPattern ceil.
        /// </summary>
        /// <param name="expr">Source expression.</param>
        /// <returns>Result expression.</returns>
        public static CallPattern Ceil(Pattern expr) => Unary(expr, UnaryOp.Ceil);

        /// <summary>
        /// CallPattern cos.
        /// </summary>
        /// <param name="expr">Source expression.</param>
        /// <returns>Result expression.</returns>
        public static CallPattern Cos(Pattern expr) => Unary(expr, UnaryOp.Cos);

        /// <summary>
        /// CallPattern exp.
        /// </summary>
        /// <param name="expr">Source expression.</param>
        /// <returns>Result expression.</returns>
        public static CallPattern Exp(Pattern expr) => Unary(expr, UnaryOp.Exp);

        /// <summary>
        /// CallPattern floor.
        /// </summary>
        /// <param name="expr">Source expression.</param>
        /// <returns>Result expression.</returns>
        public static CallPattern Floor(Pattern expr) => Unary(expr, UnaryOp.Floor);

        /// <summary>
        /// CallPattern log.
        /// </summary>
        /// <param name="expr">Source expression.</param>
        /// <returns>Result expression.</returns>
        public static CallPattern Log(Pattern expr) => Unary(expr, UnaryOp.Log);

        /// <summary>
        /// CallPattern neg.
        /// </summary>
        /// <param name="expr">Source expression.</param>
        /// <returns>Result expression.</returns>
        public static CallPattern Neg(Pattern expr) => Unary(expr, UnaryOp.Neg);

        /// <summary>
        /// CallPattern round.
        /// </summary>
        /// <param name="expr">Source expression.</param>
        /// <returns>Result expression.</returns>
        public static CallPattern Round(Pattern expr) => Unary(expr, UnaryOp.Round);

        /// <summary>
        /// CallPattern rsqrt.
        /// </summary>
        /// <param name="expr">Source expression.</param>
        /// <returns>Result expression.</returns>
        public static CallPattern Rsqrt(Pattern expr) => Unary(expr, UnaryOp.Rsqrt);

        /// <summary>
        /// CallPattern sin.
        /// </summary>
        /// <param name="expr">Source expression.</param>
        /// <returns>Result expression.</returns>
        public static CallPattern Sin(Pattern expr) => Unary(expr, UnaryOp.Sin);

        /// <summary>
        /// CallPattern sqrt.
        /// </summary>
        /// <param name="expr">Source expression.</param>
        /// <returns>Result expression.</returns>
        public static CallPattern Sqrt(Pattern expr) => Unary(expr, UnaryOp.Sqrt);

        /// <summary>
        /// CallPattern square.
        /// </summary>
        /// <param name="expr">Source expression.</param>
        /// <returns>Result expression.</returns>
        public static CallPattern Square(Pattern expr) => Unary(expr, UnaryOp.Square);

        /// <summary>
        /// CallPattern tanh.
        /// </summary>
        /// <param name="expr">Source expression.</param>
        /// <returns>Result expression.</returns>
        public static CallPattern Tanh(Pattern expr) => Unary(expr, UnaryOp.Tanh);

        /// <summary>
        /// CallPattern bitwise not.
        /// </summary>
        /// <param name="expr">Source expression.</param>
        /// <returns>Result expression.</returns>
        public static CallPattern BitwiseNot(Pattern expr) => Unary(expr, UnaryOp.BitwiseNot);

        /// <summary>
        /// CallPattern logical not.
        /// </summary>
        /// <param name="expr">Source expression.</param>
        /// <returns>Result expression.</returns>
        public static CallPattern LogicalNot(Pattern expr) => Unary(expr, UnaryOp.LogicalNot);

        /// <summary>
        /// CallPattern add.
        /// </summary>
        /// <param name="lhs">Left operand.</param>
        /// <param name="rhs">Right operand.</param>
        /// <returns>Result expression.</returns>
        public static CallPattern Add(Pattern lhs, Pattern rhs) => Binary(lhs, rhs, BinaryOp.Add);

        /// <summary>
        /// CallPattern sub.
        /// </summary>
        /// <param name="lhs">Left operand.</param>
        /// <param name="rhs">Right operand.</param>
        /// <returns>Result expression.</returns>
        public static CallPattern Sub(Pattern lhs, Pattern rhs) => Binary(lhs, rhs, BinaryOp.Sub);

        /// <summary>
        /// CallPattern mul.
        /// </summary>
        /// <param name="lhs">Left operand.</param>
        /// <param name="rhs">Right operand.</param>
        /// <returns>Result expression.</returns>
        public static CallPattern Mul(Pattern lhs, Pattern rhs) => Binary(lhs, rhs, BinaryOp.Mul);

        /// <summary>
        /// CallPattern div.
        /// </summary>
        /// <param name="lhs">Left operand.</param>
        /// <param name="rhs">Right operand.</param>
        /// <returns>Result expression.</returns>
        public static CallPattern Div(Pattern lhs, Pattern rhs) => Binary(lhs, rhs, BinaryOp.Div);

        /// <summary>
        /// CallPattern mod.
        /// </summary>
        /// <param name="lhs">Left operand.</param>
        /// <param name="rhs">Right operand.</param>
        /// <returns>Result expression.</returns>
        public static CallPattern Mod(Pattern lhs, Pattern rhs) => Binary(lhs, rhs, BinaryOp.Mod);

        /// <summary>
        /// CallPattern min.
        /// </summary>
        /// <param name="lhs">Left operand.</param>
        /// <param name="rhs">Right operand.</param>
        /// <returns>Result expression.</returns>
        public static CallPattern Min(Pattern lhs, Pattern rhs) => Binary(lhs, rhs, BinaryOp.Min);

        /// <summary>
        /// CallPattern max.
        /// </summary>
        /// <param name="lhs">Left operand.</param>
        /// <param name="rhs">Right operand.</param>
        /// <returns>Result expression.</returns>
        public static CallPattern Max(Pattern lhs, Pattern rhs) => Binary(lhs, rhs, BinaryOp.Max);

        /// <summary>
        /// CallPattern pow.
        /// </summary>
        /// <param name="lhs">Left operand.</param>
        /// <param name="rhs">Right operand.</param>
        /// <returns>Result expression.</returns>
        public static CallPattern Pow(Pattern lhs, Pattern rhs) => Binary(lhs, rhs, BinaryOp.Pow);

        /// <summary>
        /// CallPattern bitwise and.
        /// </summary>
        /// <param name="lhs">Left operand.</param>
        /// <param name="rhs">Right operand.</param>
        /// <returns>Result expression.</returns>
        public static CallPattern BitwiseAnd(Pattern lhs, Pattern rhs) => Binary(lhs, rhs, BinaryOp.BitwiseAnd);

        /// <summary>
        /// CallPattern bitwise or.
        /// </summary>
        /// <param name="lhs">Left operand.</param>
        /// <param name="rhs">Right operand.</param>
        /// <returns>Result expression.</returns>
        public static CallPattern BitwiseOr(Pattern lhs, Pattern rhs) => Binary(lhs, rhs, BinaryOp.BitwiseOr);

        /// <summary>
        /// CallPattern bitwise xor.
        /// </summary>
        /// <param name="lhs">Left operand.</param>
        /// <param name="rhs">Right operand.</param>
        /// <returns>Result expression.</returns>
        public static CallPattern BitwiseXor(Pattern lhs, Pattern rhs) => Binary(lhs, rhs, BinaryOp.BitwiseXor);

        /// <summary>
        /// CallPattern logical and.
        /// </summary>
        /// <param name="lhs">Left operand.</param>
        /// <param name="rhs">Right operand.</param>
        /// <returns>Result expression.</returns>
        public static CallPattern LogicalAnd(Pattern lhs, Pattern rhs) => Binary(lhs, rhs, BinaryOp.LogicalAnd);

        /// <summary>
        /// CallPattern logical or.
        /// </summary>
        /// <param name="lhs">Left operand.</param>
        /// <param name="rhs">Right operand.</param>
        /// <returns>Result expression.</returns>
        public static CallPattern LogicalOr(Pattern lhs, Pattern rhs) => Binary(lhs, rhs, BinaryOp.LogicalOr);

        /// <summary>
        /// CallPattern logical xor.
        /// </summary>
        /// <param name="lhs">Left operand.</param>
        /// <param name="rhs">Right operand.</param>
        /// <returns>Result expression.</returns>
        public static CallPattern LogicalXor(Pattern lhs, Pattern rhs) => Binary(lhs, rhs, BinaryOp.LogicalXor);

        /// <summary>
        /// CallPattern left shift.
        /// </summary>
        /// <param name="lhs">Left operand.</param>
        /// <param name="rhs">Right operand.</param>
        /// <returns>Result expression.</returns>
        public static CallPattern LeftShift(Pattern lhs, Pattern rhs) => Binary(lhs, rhs, BinaryOp.LeftShift);

        /// <summary>
        /// CallPattern right shift.
        /// </summary>
        /// <param name="lhs">Left operand.</param>
        /// <param name="rhs">Right operand.</param>
        /// <returns>Result expression.</returns>
        public static CallPattern RightShift(Pattern lhs, Pattern rhs) => Binary(lhs, rhs, BinaryOp.RightShift);

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
        public static CallPattern FloorMod(Pattern lhs, Pattern rhs) => Sub(lhs, (FloorDiv(lhs, rhs) * rhs));

        /// <summary>
        /// CallPattern equal.
        /// </summary>
        /// <param name="lhs"></param>
        /// <param name="rhs"></param>
        /// <returns>result.</returns>
        public static CallPattern Equal(Pattern lhs, Pattern rhs) => Compare(lhs, rhs, CompareOp.Equal);

        /// <summary>
        /// CallPattern not equal.
        /// </summary>
        /// <param name="lhs"></param>
        /// <param name="rhs"></param>
        /// <returns></returns>
        public static CallPattern NotEqual(Pattern lhs, Pattern rhs) => Compare(lhs, rhs, CompareOp.NotEqual);

        /// <summary>
        /// CallPattern less than.
        /// </summary>
        /// <param name="lhs"></param>
        /// <param name="rhs"></param>
        /// <returns></returns>
        public static CallPattern LessThan(Pattern lhs, Pattern rhs) => Compare(lhs, rhs, CompareOp.LowerThan);

        /// <summary>
        /// CallPattern less equal.
        /// </summary>
        /// <param name="lhs"></param>
        /// <param name="rhs"></param>
        /// <returns></returns>
        public static CallPattern LessEqual(Pattern lhs, Pattern rhs) => Compare(lhs, rhs, CompareOp.LowerOrEqual);

        /// <summary>
        /// CallPattern greater equal.
        /// </summary>
        /// <param name="lhs"></param>
        /// <param name="rhs"></param>
        /// <returns></returns>
        public static CallPattern GreaterEqual(Pattern lhs, Pattern rhs) => Compare(lhs, rhs, CompareOp.GreaterThan);

        /// <summary>
        /// CallPattern greater than.
        /// </summary>
        /// <param name="lhs"></param>
        /// <param name="rhs"></param>
        /// <returns></returns>
        public static CallPattern GreaterThan(Pattern lhs, Pattern rhs) => Compare(lhs, rhs, CompareOp.GreaterOrEqual);
    }
}
