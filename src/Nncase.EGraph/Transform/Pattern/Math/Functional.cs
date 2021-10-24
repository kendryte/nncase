// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.Transform.Pattern.Math;
using Nncase.IR;

namespace Nncase.Transform.Pattern
{
    public static partial class Functional
    {
        /// <summary>
        /// Math functional helper.
        /// </summary>
        public static class Math
        {
            /// <summary>
            /// CallPattern unary.
            /// </summary>
            /// <param name="unaryOp">Unary operator.</param>
            /// <param name="expr">Source expression.</param>
            /// <returns>Result expression.</returns>
            public static CallPattern Unary(UnaryOp unaryOp, ExprPattern expr)
            {
                return new CallPattern(new UnaryPattern(unaryOp), expr);
            }

            /// <summary>
            /// CallPattern binary.
            /// </summary>
            /// <param name="binaryOp">Binary operator.</param>
            /// <param name="lhs">Left operand.</param>
            /// <param name="rhs">Right operand.</param>
            /// <returns>Result expression.</returns>
            public static CallPattern Binary(BinaryOp binaryOp, ExprPattern lhs, ExprPattern rhs)
            {
                return new CallPattern(new BinaryPattern(binaryOp), lhs, rhs);
            }

            /// <summary>
            /// CallPattern clamp.
            /// </summary>
            /// <param name="input">Input expression.</param>
            /// <param name="min">Left operand.</param>
            /// <param name="max">Right operand.</param>
            /// <returns>Result expression.</returns>
            public static CallPattern Clamp(ExprPattern input, ExprPattern min, ExprPattern max)
            {
                return new CallPattern(new ClampPattern(x => true), input, min, max);
            }

            /// <summary>
            /// CallPattern clamp.
            /// </summary>
            /// <param name="input">Input expression.</param>
            /// <param name="range">Value range.</param>
            /// <typeparam name="T">Data type.</typeparam>
            /// <returns>Result expression.</returns>
            public static CallPattern Clamp<T>(ExprPattern input, ValueRange<T> range)
                where T : unmanaged
            {
                return new CallPattern(new ClampPattern(x => true), input, new ConstPattern(Const.FromScalar(range.Min)), new ConstPattern(Const.FromScalar(range.Max)));
            }

            /// <summary>
            /// CallPattern abs.
            /// </summary>
            /// <param name="expr">Source expression.</param>
            /// <returns>Result expression.</returns>
            public static CallPattern Abs(ExprPattern expr) => Unary(UnaryOp.Abs, expr);

            /// <summary>
            /// CallPattern ceil.
            /// </summary>
            /// <param name="expr">Source expression.</param>
            /// <returns>Result expression.</returns>
            public static CallPattern Ceil(ExprPattern expr) => Unary(UnaryOp.Ceil, expr);

            /// <summary>
            /// CallPattern cos.
            /// </summary>
            /// <param name="expr">Source expression.</param>
            /// <returns>Result expression.</returns>
            public static CallPattern Cos(ExprPattern expr) => Unary(UnaryOp.Cos, expr);

            /// <summary>
            /// CallPattern exp.
            /// </summary>
            /// <param name="expr">Source expression.</param>
            /// <returns>Result expression.</returns>
            public static CallPattern Exp(ExprPattern expr) => Unary(UnaryOp.Exp, expr);

            /// <summary>
            /// CallPattern floor.
            /// </summary>
            /// <param name="expr">Source expression.</param>
            /// <returns>Result expression.</returns>
            public static CallPattern Floor(ExprPattern expr) => Unary(UnaryOp.Floor, expr);

            /// <summary>
            /// CallPattern log.
            /// </summary>
            /// <param name="expr">Source expression.</param>
            /// <returns>Result expression.</returns>
            public static CallPattern Log(ExprPattern expr) => Unary(UnaryOp.Log, expr);

            /// <summary>
            /// CallPattern neg.
            /// </summary>
            /// <param name="expr">Source expression.</param>
            /// <returns>Result expression.</returns>
            public static CallPattern Neg(ExprPattern expr) => Unary(UnaryOp.Neg, expr);

            /// <summary>
            /// CallPattern round.
            /// </summary>
            /// <param name="expr">Source expression.</param>
            /// <returns>Result expression.</returns>
            public static CallPattern Round(ExprPattern expr) => Unary(UnaryOp.Round, expr);

            /// <summary>
            /// CallPattern rsqrt.
            /// </summary>
            /// <param name="expr">Source expression.</param>
            /// <returns>Result expression.</returns>
            public static CallPattern Rsqrt(ExprPattern expr) => Unary(UnaryOp.Rsqrt, expr);

            /// <summary>
            /// CallPattern sin.
            /// </summary>
            /// <param name="expr">Source expression.</param>
            /// <returns>Result expression.</returns>
            public static CallPattern Sin(ExprPattern expr) => Unary(UnaryOp.Sin, expr);

            /// <summary>
            /// CallPattern sqrt.
            /// </summary>
            /// <param name="expr">Source expression.</param>
            /// <returns>Result expression.</returns>
            public static CallPattern Sqrt(ExprPattern expr) => Unary(UnaryOp.Sqrt, expr);

            /// <summary>
            /// CallPattern square.
            /// </summary>
            /// <param name="expr">Source expression.</param>
            /// <returns>Result expression.</returns>
            public static CallPattern Square(ExprPattern expr) => Unary(UnaryOp.Square, expr);

            /// <summary>
            /// CallPattern tanh.
            /// </summary>
            /// <param name="expr">Source expression.</param>
            /// <returns>Result expression.</returns>
            public static CallPattern Tanh(ExprPattern expr) => Unary(UnaryOp.Tanh, expr);

            /// <summary>
            /// CallPattern bitwise not.
            /// </summary>
            /// <param name="expr">Source expression.</param>
            /// <returns>Result expression.</returns>
            public static CallPattern BitwiseNot(ExprPattern expr) => Unary(UnaryOp.BitwiseNot, expr);

            /// <summary>
            /// CallPattern logical not.
            /// </summary>
            /// <param name="expr">Source expression.</param>
            /// <returns>Result expression.</returns>
            public static CallPattern LogicalNot(ExprPattern expr) => Unary(UnaryOp.LogicalNot, expr);

            /// <summary>
            /// CallPattern add.
            /// </summary>
            /// <param name="lhs">Left operand.</param>
            /// <param name="rhs">Right operand.</param>
            /// <returns>Result expression.</returns>
            public static CallPattern Add(ExprPattern lhs, ExprPattern rhs) => Binary(BinaryOp.Add, lhs, rhs);

            /// <summary>
            /// CallPattern sub.
            /// </summary>
            /// <param name="lhs">Left operand.</param>
            /// <param name="rhs">Right operand.</param>
            /// <returns>Result expression.</returns>
            public static CallPattern Sub(ExprPattern lhs, ExprPattern rhs) => Binary(BinaryOp.Sub, lhs, rhs);

            /// <summary>
            /// CallPattern mul.
            /// </summary>
            /// <param name="lhs">Left operand.</param>
            /// <param name="rhs">Right operand.</param>
            /// <returns>Result expression.</returns>
            public static CallPattern Mul(ExprPattern lhs, ExprPattern rhs) => Binary(BinaryOp.Mul, lhs, rhs);

            /// <summary>
            /// CallPattern div.
            /// </summary>
            /// <param name="lhs">Left operand.</param>
            /// <param name="rhs">Right operand.</param>
            /// <returns>Result expression.</returns>
            public static CallPattern Div(ExprPattern lhs, ExprPattern rhs) => Binary(BinaryOp.Div, lhs, rhs);

            /// <summary>
            /// CallPattern mod.
            /// </summary>
            /// <param name="lhs">Left operand.</param>
            /// <param name="rhs">Right operand.</param>
            /// <returns>Result expression.</returns>
            public static CallPattern Mod(ExprPattern lhs, ExprPattern rhs) => Binary(BinaryOp.Mod, lhs, rhs);

            /// <summary>
            /// CallPattern min.
            /// </summary>
            /// <param name="lhs">Left operand.</param>
            /// <param name="rhs">Right operand.</param>
            /// <returns>Result expression.</returns>
            public static CallPattern Min(ExprPattern lhs, ExprPattern rhs) => Binary(BinaryOp.Min, lhs, rhs);

            /// <summary>
            /// CallPattern max.
            /// </summary>
            /// <param name="lhs">Left operand.</param>
            /// <param name="rhs">Right operand.</param>
            /// <returns>Result expression.</returns>
            public static CallPattern Max(ExprPattern lhs, ExprPattern rhs) => Binary(BinaryOp.Max, lhs, rhs);

            /// <summary>
            /// CallPattern pow.
            /// </summary>
            /// <param name="lhs">Left operand.</param>
            /// <param name="rhs">Right operand.</param>
            /// <returns>Result expression.</returns>
            public static CallPattern Pow(ExprPattern lhs, ExprPattern rhs) => Binary(BinaryOp.Pow, lhs, rhs);

            /// <summary>
            /// CallPattern bitwise and.
            /// </summary>
            /// <param name="lhs">Left operand.</param>
            /// <param name="rhs">Right operand.</param>
            /// <returns>Result expression.</returns>
            public static CallPattern BitwiseAnd(ExprPattern lhs, ExprPattern rhs) => Binary(BinaryOp.BitwiseAnd, lhs, rhs);

            /// <summary>
            /// CallPattern bitwise or.
            /// </summary>
            /// <param name="lhs">Left operand.</param>
            /// <param name="rhs">Right operand.</param>
            /// <returns>Result expression.</returns>
            public static CallPattern BitwiseOr(ExprPattern lhs, ExprPattern rhs) => Binary(BinaryOp.BitwiseOr, lhs, rhs);

            /// <summary>
            /// CallPattern bitwise xor.
            /// </summary>
            /// <param name="lhs">Left operand.</param>
            /// <param name="rhs">Right operand.</param>
            /// <returns>Result expression.</returns>
            public static CallPattern BitwiseXor(ExprPattern lhs, ExprPattern rhs) => Binary(BinaryOp.BitwiseXor, lhs, rhs);

            /// <summary>
            /// CallPattern logical and.
            /// </summary>
            /// <param name="lhs">Left operand.</param>
            /// <param name="rhs">Right operand.</param>
            /// <returns>Result expression.</returns>
            public static CallPattern LogicalAnd(ExprPattern lhs, ExprPattern rhs) => Binary(BinaryOp.LogicalAnd, lhs, rhs);

            /// <summary>
            /// CallPattern logical or.
            /// </summary>
            /// <param name="lhs">Left operand.</param>
            /// <param name="rhs">Right operand.</param>
            /// <returns>Result expression.</returns>
            public static CallPattern LogicalOr(ExprPattern lhs, ExprPattern rhs) => Binary(BinaryOp.LogicalOr, lhs, rhs);

            /// <summary>
            /// CallPattern logical xor.
            /// </summary>
            /// <param name="lhs">Left operand.</param>
            /// <param name="rhs">Right operand.</param>
            /// <returns>Result expression.</returns>
            public static CallPattern LogicalXor(ExprPattern lhs, ExprPattern rhs) => Binary(BinaryOp.LogicalXor, lhs, rhs);

            /// <summary>
            /// CallPattern floor div.
            /// </summary>
            /// <param name="lhs">Left operand.</param>
            /// <param name="rhs">Right operand.</param>
            /// <returns>Result expression.</returns>
            public static CallPattern FloorDiv(ExprPattern lhs, ExprPattern rhs) => Floor(lhs / rhs);

            /// <summary>
            /// CallPattern floor mod.
            /// </summary>
            /// <param name="lhs">Left operand.</param>
            /// <param name="rhs">Right operand.</param>
            /// <returns>Result expression.</returns>
            public static CallPattern FloorMod(ExprPattern lhs, ExprPattern rhs) => lhs - (FloorDiv(lhs, rhs) * rhs);
        }
    }
}
