// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.Transform.Pattern.Math;
using Nncase.IR;

namespace Nncase.Transform.Pattern.F
{
    public static class Math
    {

        public static CallPattern Unary(UnaryOp unaryOp, ExprPattern expr)
        {
            return new CallPattern(new UnaryPattern(unaryOp), expr);
        }



        public static CallPattern Binary(BinaryOp binaryOp, ExprPattern lhs, ExprPattern rhs)
        {
            return new CallPattern(new BinaryPattern(binaryOp), lhs, rhs);
        }



        public static CallPattern Clamp(ExprPattern input, ExprPattern min, ExprPattern max)
        {
            return new CallPattern(new ClampPattern(x => true), input, min, max);
        }



        public static CallPattern Clamp<T>(ExprPattern input, ValueRange<T> range)
            where T : unmanaged
        {
            return new CallPattern(new ClampPattern(x => true),
             input,
             (ConstPattern)Const.FromScalar(range.Min),
             (ConstPattern)Const.FromScalar(range.Max));
        }


        public static CallPattern Abs(ExprPattern expr) => Unary(UnaryOp.Abs, expr);

        public static CallPattern Ceil(ExprPattern expr) => Unary(UnaryOp.Ceil, expr);

        public static CallPattern Cos(ExprPattern expr) => Unary(UnaryOp.Cos, expr);

        public static CallPattern Exp(ExprPattern expr) => Unary(UnaryOp.Exp, expr);

        public static CallPattern Floor(ExprPattern expr) => Unary(UnaryOp.Floor, expr);

        public static CallPattern Log(ExprPattern expr) => Unary(UnaryOp.Log, expr);

        public static CallPattern Neg(ExprPattern expr) => Unary(UnaryOp.Neg, expr);

        public static CallPattern Round(ExprPattern expr) => Unary(UnaryOp.Round, expr);

        public static CallPattern Rsqrt(ExprPattern expr) => Unary(UnaryOp.Rsqrt, expr);

        public static CallPattern Sin(ExprPattern expr) => Unary(UnaryOp.Sin, expr);

        public static CallPattern Sqrt(ExprPattern expr) => Unary(UnaryOp.Sqrt, expr);

        public static CallPattern Square(ExprPattern expr) => Unary(UnaryOp.Square, expr);

        public static CallPattern Tanh(ExprPattern expr) => Unary(UnaryOp.Tanh, expr);

        public static CallPattern BitwiseNot(ExprPattern expr) => Unary(UnaryOp.BitwiseNot, expr);

        public static CallPattern LogicalNot(ExprPattern expr) => Unary(UnaryOp.LogicalNot, expr);
        public static CallPattern Add(ExprPattern lhs, ExprPattern rhs) => Binary(BinaryOp.Add, lhs, rhs);

        public static CallPattern Sub(ExprPattern lhs, ExprPattern rhs) => Binary(BinaryOp.Sub, lhs, rhs);

        public static CallPattern Mul(ExprPattern lhs, ExprPattern rhs) => Binary(BinaryOp.Mul, lhs, rhs);

        public static CallPattern Div(ExprPattern lhs, ExprPattern rhs) => Binary(BinaryOp.Div, lhs, rhs);

        public static CallPattern Mod(ExprPattern lhs, ExprPattern rhs) => Binary(BinaryOp.Mod, lhs, rhs);

        public static CallPattern Min(ExprPattern lhs, ExprPattern rhs) => Binary(BinaryOp.Min, lhs, rhs);

        public static CallPattern Max(ExprPattern lhs, ExprPattern rhs) => Binary(BinaryOp.Max, lhs, rhs);

        public static CallPattern Pow(ExprPattern lhs, ExprPattern rhs) => Binary(BinaryOp.Pow, lhs, rhs);

        public static CallPattern BitwiseAnd(ExprPattern lhs, ExprPattern rhs) => Binary(BinaryOp.BitwiseAnd, lhs, rhs);

        public static CallPattern BitwiseOr(ExprPattern lhs, ExprPattern rhs) => Binary(BinaryOp.BitwiseOr, lhs, rhs);

        public static CallPattern BitwiseXor(ExprPattern lhs, ExprPattern rhs) => Binary(BinaryOp.BitwiseXor, lhs, rhs);

        public static CallPattern LogicalAnd(ExprPattern lhs, ExprPattern rhs) => Binary(BinaryOp.LogicalAnd, lhs, rhs);

        public static CallPattern LogicalOr(ExprPattern lhs, ExprPattern rhs) => Binary(BinaryOp.LogicalOr, lhs, rhs);

        public static CallPattern LogicalXor(ExprPattern lhs, ExprPattern rhs) => Binary(BinaryOp.LogicalXor, lhs, rhs);

        public static CallPattern FloorDiv(ExprPattern lhs, ExprPattern rhs) => Floor(lhs / rhs);

    }

}
