// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR.Math;

namespace Nncase.IR.F
{
    /// <summary>
    /// Math functional helper.
    /// </summary>
    public static class Math
    {
        /// <summary>
        /// Call unary.
        /// </summary>
        /// <param name="unaryOp">Unary operator.</param>
        /// <param name="expr">Source expression.</param>
        /// <returns>Result expression.</returns>
        public static Call Unary(UnaryOp unaryOp, Expr expr)
        {
            return new Call(new Unary(unaryOp), expr);
        }

        /// <summary>
        /// Call binary.
        /// </summary>
        /// <param name="binaryOp">Binary operator.</param>
        /// <param name="lhs">Left operand.</param>
        /// <param name="rhs">Right operand.</param>
        /// <returns>Result expression.</returns>
        public static Call Binary(BinaryOp binaryOp, Expr lhs, Expr rhs)
        {
            return new Call(new Binary(binaryOp), lhs, rhs);
        }

        /// <summary>
        /// Call clamp.
        /// </summary>
        /// <param name="input">Input expression.</param>
        /// <param name="min">Left operand.</param>
        /// <param name="max">Right operand.</param>
        /// <returns>Result expression.</returns>
        public static Call Clamp(Expr input, Expr min, Expr max)
        {
            return new Call(new Clamp(), input, min, max);
        }

        /// <summary>
        /// Call clamp.
        /// </summary>
        /// <param name="input">Input expression.</param>
        /// <param name="range">Value range.</param>
        /// <typeparam name="T">Data type.</typeparam>
        /// <returns>Result expression.</returns>
        public static Call Clamp<T>(Expr input, ValueRange<T> range)
            where T : unmanaged
        {
            return new Call(new Clamp(), input, Const.FromScalar(range.Min), Const.FromScalar(range.Max));
        }

        /// <summary>
        /// Call abs.
        /// </summary>
        /// <param name="expr">Source expression.</param>
        /// <returns>Result expression.</returns>
        public static Call Abs(Expr expr) => Unary(UnaryOp.Abs, expr);

        /// <summary>
        /// Call ceil.
        /// </summary>
        /// <param name="expr">Source expression.</param>
        /// <returns>Result expression.</returns>
        public static Call Ceil(Expr expr) => Unary(UnaryOp.Ceil, expr);

        /// <summary>
        /// Call cos.
        /// </summary>
        /// <param name="expr">Source expression.</param>
        /// <returns>Result expression.</returns>
        public static Call Cos(Expr expr) => Unary(UnaryOp.Cos, expr);

        /// <summary>
        /// Call exp.
        /// </summary>
        /// <param name="expr">Source expression.</param>
        /// <returns>Result expression.</returns>
        public static Call Exp(Expr expr) => Unary(UnaryOp.Exp, expr);

        /// <summary>
        /// Call floor.
        /// </summary>
        /// <param name="expr">Source expression.</param>
        /// <returns>Result expression.</returns>
        public static Call Floor(Expr expr) => Unary(UnaryOp.Floor, expr);

        /// <summary>
        /// Call log.
        /// </summary>
        /// <param name="expr">Source expression.</param>
        /// <returns>Result expression.</returns>
        public static Call Log(Expr expr) => Unary(UnaryOp.Log, expr);

        /// <summary>
        /// Call neg.
        /// </summary>
        /// <param name="expr">Source expression.</param>
        /// <returns>Result expression.</returns>
        public static Call Neg(Expr expr) => Unary(UnaryOp.Neg, expr);

        /// <summary>
        /// Call round.
        /// </summary>
        /// <param name="expr">Source expression.</param>
        /// <returns>Result expression.</returns>
        public static Call Round(Expr expr) => Unary(UnaryOp.Round, expr);

        /// <summary>
        /// Call rsqrt.
        /// </summary>
        /// <param name="expr">Source expression.</param>
        /// <returns>Result expression.</returns>
        public static Call Rsqrt(Expr expr) => Unary(UnaryOp.Rsqrt, expr);

        /// <summary>
        /// Call sin.
        /// </summary>
        /// <param name="expr">Source expression.</param>
        /// <returns>Result expression.</returns>
        public static Call Sin(Expr expr) => Unary(UnaryOp.Sin, expr);

        /// <summary>
        /// Call sqrt.
        /// </summary>
        /// <param name="expr">Source expression.</param>
        /// <returns>Result expression.</returns>
        public static Call Sqrt(Expr expr) => Unary(UnaryOp.Sqrt, expr);

        /// <summary>
        /// Call square.
        /// </summary>
        /// <param name="expr">Source expression.</param>
        /// <returns>Result expression.</returns>
        public static Call Square(Expr expr) => Unary(UnaryOp.Square, expr);

        /// <summary>
        /// Call tanh.
        /// </summary>
        /// <param name="expr">Source expression.</param>
        /// <returns>Result expression.</returns>
        public static Call Tanh(Expr expr) => Unary(UnaryOp.Tanh, expr);

        /// <summary>
        /// Call bitwise not.
        /// </summary>
        /// <param name="expr">Source expression.</param>
        /// <returns>Result expression.</returns>
        public static Call BitwiseNot(Expr expr) => Unary(UnaryOp.BitwiseNot, expr);

        /// <summary>
        /// Call logical not.
        /// </summary>
        /// <param name="expr">Source expression.</param>
        /// <returns>Result expression.</returns>
        public static Call LogicalNot(Expr expr) => Unary(UnaryOp.LogicalNot, expr);

        /// <summary>
        /// Call add.
        /// </summary>
        /// <param name="lhs">Left operand.</param>
        /// <param name="rhs">Right operand.</param>
        /// <returns>Result expression.</returns>
        public static Call Add(Expr lhs, Expr rhs) => Binary(BinaryOp.Add, lhs, rhs);

        /// <summary>
        /// Call sub.
        /// </summary>
        /// <param name="lhs">Left operand.</param>
        /// <param name="rhs">Right operand.</param>
        /// <returns>Result expression.</returns>
        public static Call Sub(Expr lhs, Expr rhs) => Binary(BinaryOp.Sub, lhs, rhs);

        /// <summary>
        /// Call mul.
        /// </summary>
        /// <param name="lhs">Left operand.</param>
        /// <param name="rhs">Right operand.</param>
        /// <returns>Result expression.</returns>
        public static Call Mul(Expr lhs, Expr rhs) => Binary(BinaryOp.Mul, lhs, rhs);

        /// <summary>
        /// Call div.
        /// </summary>
        /// <param name="lhs">Left operand.</param>
        /// <param name="rhs">Right operand.</param>
        /// <returns>Result expression.</returns>
        public static Call Div(Expr lhs, Expr rhs) => Binary(BinaryOp.Div, lhs, rhs);

        /// <summary>
        /// Call mod.
        /// </summary>
        /// <param name="lhs">Left operand.</param>
        /// <param name="rhs">Right operand.</param>
        /// <returns>Result expression.</returns>
        public static Call Mod(Expr lhs, Expr rhs) => Binary(BinaryOp.Mod, lhs, rhs);

        /// <summary>
        /// Call min.
        /// </summary>
        /// <param name="lhs">Left operand.</param>
        /// <param name="rhs">Right operand.</param>
        /// <returns>Result expression.</returns>
        public static Call Min(Expr lhs, Expr rhs) => Binary(BinaryOp.Min, lhs, rhs);

        /// <summary>
        /// Call max.
        /// </summary>
        /// <param name="lhs">Left operand.</param>
        /// <param name="rhs">Right operand.</param>
        /// <returns>Result expression.</returns>
        public static Call Max(Expr lhs, Expr rhs) => Binary(BinaryOp.Max, lhs, rhs);

        /// <summary>
        /// Call pow.
        /// </summary>
        /// <param name="lhs">Left operand.</param>
        /// <param name="rhs">Right operand.</param>
        /// <returns>Result expression.</returns>
        public static Call Pow(Expr lhs, Expr rhs) => Binary(BinaryOp.Pow, lhs, rhs);

        /// <summary>
        /// Call bitwise and.
        /// </summary>
        /// <param name="lhs">Left operand.</param>
        /// <param name="rhs">Right operand.</param>
        /// <returns>Result expression.</returns>
        public static Call BitwiseAnd(Expr lhs, Expr rhs) => Binary(BinaryOp.BitwiseAnd, lhs, rhs);

        /// <summary>
        /// Call bitwise or.
        /// </summary>
        /// <param name="lhs">Left operand.</param>
        /// <param name="rhs">Right operand.</param>
        /// <returns>Result expression.</returns>
        public static Call BitwiseOr(Expr lhs, Expr rhs) => Binary(BinaryOp.BitwiseOr, lhs, rhs);

        /// <summary>
        /// Call bitwise xor.
        /// </summary>
        /// <param name="lhs">Left operand.</param>
        /// <param name="rhs">Right operand.</param>
        /// <returns>Result expression.</returns>
        public static Call BitwiseXor(Expr lhs, Expr rhs) => Binary(BinaryOp.BitwiseXor, lhs, rhs);

        /// <summary>
        /// Call logical and.
        /// </summary>
        /// <param name="lhs">Left operand.</param>
        /// <param name="rhs">Right operand.</param>
        /// <returns>Result expression.</returns>
        public static Call LogicalAnd(Expr lhs, Expr rhs) => Binary(BinaryOp.LogicalAnd, lhs, rhs);

        /// <summary>
        /// Call logical or.
        /// </summary>
        /// <param name="lhs">Left operand.</param>
        /// <param name="rhs">Right operand.</param>
        /// <returns>Result expression.</returns>
        public static Call LogicalOr(Expr lhs, Expr rhs) => Binary(BinaryOp.LogicalOr, lhs, rhs);

        /// <summary>
        /// Call logical xor.
        /// </summary>
        /// <param name="lhs">Left operand.</param>
        /// <param name="rhs">Right operand.</param>
        /// <returns>Result expression.</returns>
        public static Call LogicalXor(Expr lhs, Expr rhs) => Binary(BinaryOp.LogicalXor, lhs, rhs);

        /// <summary>
        /// Call floor div.
        /// </summary>
        /// <param name="lhs">Left operand.</param>
        /// <param name="rhs">Right operand.</param>
        /// <returns>Result expression.</returns>
        public static Call FloorDiv(Expr lhs, Expr rhs) => Floor(lhs / rhs);

        /// <summary>
        /// Call floor mod.
        /// </summary>
        /// <param name="lhs">Left operand.</param>
        /// <param name="rhs">Right operand.</param>
        /// <returns>Result expression.</returns>
        public static Call FloorMod(Expr lhs, Expr rhs) => Sub(lhs, (FloorDiv(lhs, rhs) * rhs));
    }
}
