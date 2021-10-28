// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.Transform.Pattern.Math;
using static Nncase.Transform.Pattern.Utility;
using Nncase.IR;

namespace Nncase.Transform.Pattern.F
{
    public static class Math
    {

        public static CallPattern Unary(UnaryOp unaryOp, ExprPattern expr) => Unary(GetID(), unaryOp, expr);
        public static CallPattern Unary(ID Id, UnaryOp unaryOp, ExprPattern expr)
        {
            return new CallPattern(Id, new UnaryPattern(unaryOp), expr);
        }


        public static CallPattern Binary(BinaryOp binaryOp, ExprPattern lhs, ExprPattern rhs) => Binary(GetID(), binaryOp, lhs, rhs);

        public static CallPattern Binary(ID Id, BinaryOp binaryOp, ExprPattern lhs, ExprPattern rhs)
        {
            return new CallPattern(Id, new BinaryPattern(binaryOp), lhs, rhs);
        }


        public static CallPattern Clamp(ExprPattern input, ExprPattern min, ExprPattern max) => Clamp(GetID(), input, min, max);

        public static CallPattern Clamp(ID Id, ExprPattern input, ExprPattern min, ExprPattern max)
        {
            return new CallPattern(Id, new ClampPattern(x => true), input, min, max);
        }


        public static CallPattern Clamp<T>(ExprPattern input, ValueRange<T> range) where T : unmanaged => Clamp<T>(GetID(), input, range);

        public static CallPattern Clamp<T>(ID Id, ExprPattern input, ValueRange<T> range)
            where T : unmanaged
        {
            return new CallPattern(Id, new ClampPattern(x => true),
             input,
             (ConstPattern)Const.FromScalar(range.Min),
             (ConstPattern)Const.FromScalar(range.Max));
        }


        public static CallPattern Abs(ID Id, ExprPattern expr) => Unary(Id, UnaryOp.Abs, expr);
        public static CallPattern Abs(ExprPattern expr) => Abs(GetID(), expr);

        public static CallPattern Ceil(ID Id, ExprPattern expr) => Unary(Id, UnaryOp.Ceil, expr);
        public static CallPattern Ceil(ExprPattern expr) => Ceil(GetID(), expr);

        public static CallPattern Cos(ID Id, ExprPattern expr) => Unary(Id, UnaryOp.Cos, expr);
        public static CallPattern Cos(ExprPattern expr) => Cos(GetID(), expr);

        public static CallPattern Exp(ID Id, ExprPattern expr) => Unary(Id, UnaryOp.Exp, expr);
        public static CallPattern Exp(ExprPattern expr) => Exp(GetID(), expr);

        public static CallPattern Floor(ID Id, ExprPattern expr) => Unary(Id, UnaryOp.Floor, expr);
        public static CallPattern Floor(ExprPattern expr) => Floor(GetID(), expr);

        public static CallPattern Log(ID Id, ExprPattern expr) => Unary(Id, UnaryOp.Log, expr);
        public static CallPattern Log(ExprPattern expr) => Log(GetID(), expr);

        public static CallPattern Neg(ID Id, ExprPattern expr) => Unary(Id, UnaryOp.Neg, expr);
        public static CallPattern Neg(ExprPattern expr) => Neg(GetID(), expr);

        public static CallPattern Round(ID Id, ExprPattern expr) => Unary(Id, UnaryOp.Round, expr);
        public static CallPattern Round(ExprPattern expr) => Round(GetID(), expr);

        public static CallPattern Rsqrt(ID Id, ExprPattern expr) => Unary(Id, UnaryOp.Rsqrt, expr);
        public static CallPattern Rsqrt(ExprPattern expr) => Rsqrt(GetID(), expr);

        public static CallPattern Sin(ID Id, ExprPattern expr) => Unary(Id, UnaryOp.Sin, expr);
        public static CallPattern Sin(ExprPattern expr) => Sin(GetID(), expr);

        public static CallPattern Sqrt(ID Id, ExprPattern expr) => Unary(Id, UnaryOp.Sqrt, expr);
        public static CallPattern Sqrt(ExprPattern expr) => Sqrt(GetID(), expr);

        public static CallPattern Square(ID Id, ExprPattern expr) => Unary(Id, UnaryOp.Square, expr);
        public static CallPattern Square(ExprPattern expr) => Square(GetID(), expr);

        public static CallPattern Tanh(ID Id, ExprPattern expr) => Unary(Id, UnaryOp.Tanh, expr);
        public static CallPattern Tanh(ExprPattern expr) => Tanh(GetID(), expr);

        public static CallPattern BitwiseNot(ID Id, ExprPattern expr) => Unary(Id, UnaryOp.BitwiseNot, expr);
        public static CallPattern BitwiseNot(ExprPattern expr) => BitwiseNot(GetID(), expr);

        public static CallPattern LogicalNot(ID Id, ExprPattern expr) => Unary(Id, UnaryOp.LogicalNot, expr);
        public static CallPattern LogicalNot(ExprPattern expr) => LogicalNot(GetID(), expr);
        public static CallPattern Add(ExprPattern lhs, ExprPattern rhs) => Add(GetID(), lhs, rhs);
        public static CallPattern Add(ID Id, ExprPattern lhs, ExprPattern rhs) => Binary(Id, BinaryOp.Add, lhs, rhs);

        public static CallPattern Sub(ExprPattern lhs, ExprPattern rhs) => Sub(GetID(), lhs, rhs);
        public static CallPattern Sub(ID Id, ExprPattern lhs, ExprPattern rhs) => Binary(Id, BinaryOp.Sub, lhs, rhs);

        public static CallPattern Mul(ExprPattern lhs, ExprPattern rhs) => Mul(GetID(), lhs, rhs);
        public static CallPattern Mul(ID Id, ExprPattern lhs, ExprPattern rhs) => Binary(Id, BinaryOp.Mul, lhs, rhs);

        public static CallPattern Div(ExprPattern lhs, ExprPattern rhs) => Div(GetID(), lhs, rhs);
        public static CallPattern Div(ID Id, ExprPattern lhs, ExprPattern rhs) => Binary(Id, BinaryOp.Div, lhs, rhs);

        public static CallPattern Mod(ExprPattern lhs, ExprPattern rhs) => Mod(GetID(), lhs, rhs);
        public static CallPattern Mod(ID Id, ExprPattern lhs, ExprPattern rhs) => Binary(Id, BinaryOp.Mod, lhs, rhs);

        public static CallPattern Min(ExprPattern lhs, ExprPattern rhs) => Min(GetID(), lhs, rhs);
        public static CallPattern Min(ID Id, ExprPattern lhs, ExprPattern rhs) => Binary(Id, BinaryOp.Min, lhs, rhs);

        public static CallPattern Max(ExprPattern lhs, ExprPattern rhs) => Max(GetID(), lhs, rhs);
        public static CallPattern Max(ID Id, ExprPattern lhs, ExprPattern rhs) => Binary(Id, BinaryOp.Max, lhs, rhs);

        public static CallPattern Pow(ExprPattern lhs, ExprPattern rhs) => Pow(GetID(), lhs, rhs);
        public static CallPattern Pow(ID Id, ExprPattern lhs, ExprPattern rhs) => Binary(Id, BinaryOp.Pow, lhs, rhs);

        public static CallPattern BitwiseAnd(ExprPattern lhs, ExprPattern rhs) => BitwiseAnd(GetID(), lhs, rhs);
        public static CallPattern BitwiseAnd(ID Id, ExprPattern lhs, ExprPattern rhs) => Binary(Id, BinaryOp.BitwiseAnd, lhs, rhs);

        public static CallPattern BitwiseOr(ExprPattern lhs, ExprPattern rhs) => BitwiseOr(GetID(), lhs, rhs);
        public static CallPattern BitwiseOr(ID Id, ExprPattern lhs, ExprPattern rhs) => Binary(Id, BinaryOp.BitwiseOr, lhs, rhs);

        public static CallPattern BitwiseXor(ExprPattern lhs, ExprPattern rhs) => BitwiseXor(GetID(), lhs, rhs);
        public static CallPattern BitwiseXor(ID Id, ExprPattern lhs, ExprPattern rhs) => Binary(Id, BinaryOp.BitwiseXor, lhs, rhs);

        public static CallPattern LogicalAnd(ExprPattern lhs, ExprPattern rhs) => LogicalAnd(GetID(), lhs, rhs);
        public static CallPattern LogicalAnd(ID Id, ExprPattern lhs, ExprPattern rhs) => Binary(Id, BinaryOp.LogicalAnd, lhs, rhs);

        public static CallPattern LogicalOr(ExprPattern lhs, ExprPattern rhs) => LogicalOr(GetID(), lhs, rhs);
        public static CallPattern LogicalOr(ID Id, ExprPattern lhs, ExprPattern rhs) => Binary(Id, BinaryOp.LogicalOr, lhs, rhs);

        public static CallPattern LogicalXor(ExprPattern lhs, ExprPattern rhs) => LogicalXor(GetID(), lhs, rhs);
        public static CallPattern LogicalXor(ID Id, ExprPattern lhs, ExprPattern rhs) => Binary(Id, BinaryOp.LogicalXor, lhs, rhs);

        public static CallPattern FloorDiv(ExprPattern lhs, ExprPattern rhs) => FloorDiv(GetID(), lhs, rhs);
        public static CallPattern FloorDiv(ID Id, ExprPattern lhs, ExprPattern rhs) => Floor(Id, lhs / rhs);

        public static CallPattern FloorMod(ExprPattern lhs, ExprPattern rhs) => FloorMod(GetID(), lhs, rhs);
        public static CallPattern FloorMod(ID Id, ExprPattern lhs, ExprPattern rhs) => Sub(Id, lhs, (FloorDiv(Utility.GetID(), lhs, rhs) * rhs));
    }

}
