// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR.Math;
using Nncase.Utilities;

namespace Nncase.IR.F;

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
    /// Call min max like clamp.
    /// </summary>
    public static Call MinMax(Expr input, Expr min, Expr max)
    {
        return IR.F.Math.Min(IR.F.Math.Max(input, min), max);
    }

    /// <summary>
    /// Call clamp.
    /// </summary>
    /// <param name="input">Input expression.</param>
    /// <param name="range">Value range.</param>
    /// <typeparam name="T">Data type.</typeparam>
    /// <returns>Result expression.</returns>
    public static Call Clamp<T>(Expr input, ValueRange<T> range)
        where T : unmanaged, IEquatable<T>, IComparable<T>
    {
        return new Call(new Clamp(), input, Tensor.FromScalar(range.Min), Tensor.FromScalar(range.Max));
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
    /// Call matMul.
    /// </summary>
    /// <param name="lhs">Left operand.</param>
    /// <param name="rhs">Right operand.</param>
    /// <returns>Result expression.</returns>
    public static Call MatMul(Expr lhs, Expr rhs) => new(new MatMul(), lhs, rhs);

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
    /// Call left shift.
    /// </summary>
    /// <param name="lhs">Left operand.</param>
    /// <param name="rhs">Right operand.</param>
    /// <returns>Result expression.</returns>
    public static Call LeftShift(Expr lhs, Expr rhs) => Binary(BinaryOp.LeftShift, lhs, rhs);

    /// <summary>
    /// Call right shift.
    /// </summary>
    /// <param name="lhs">Left operand.</param>
    /// <param name="rhs">Right operand.</param>
    /// <returns>Result expression.</returns>
    public static Call RightShift(Expr lhs, Expr rhs) => Binary(BinaryOp.RightShift, lhs, rhs);

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
    public static Call FloorMod(Expr lhs, Expr rhs) => Sub(lhs, FloorDiv(lhs, rhs) * rhs);

    /// <summary>
    /// Call compare.
    /// </summary>
    /// <param name="compareOp">Compare operator.</param>
    /// <param name="lhs">Left operand.</param>
    /// <param name="rhs">Right operand.</param>
    /// <returns>Result expression.</returns>
    public static Call Compare(CompareOp compareOp, Expr lhs, Expr rhs) => new Call(new Compare(compareOp), lhs, rhs);

    /// <summary>
    /// Call equal.
    /// </summary>
    public static Call Equal(Expr lhs, Expr rhs) => Compare(CompareOp.Equal, lhs, rhs);

    /// <summary>
    /// call not equal.
    /// </summary>
    public static Call NotEqual(Expr lhs, Expr rhs) => Compare(CompareOp.NotEqual, lhs, rhs);

    /// <summary>
    /// call less than.
    /// </summary>
    public static Call LessThan(Expr lhs, Expr rhs) => Compare(CompareOp.LowerThan, lhs, rhs);

    /// <summary>
    /// call less equal.
    /// </summary>
    public static Call LessEqual(Expr lhs, Expr rhs) => Compare(CompareOp.LowerOrEqual, lhs, rhs);

    /// <summary>
    /// call greater equal.
    /// </summary>
    public static Call GreaterEqual(Expr lhs, Expr rhs) => Compare(CompareOp.GreaterOrEqual, lhs, rhs);

    /// <summary>
    /// call greater than.
    /// </summary>
    public static Call GreaterThan(Expr lhs, Expr rhs) => Compare(CompareOp.GreaterThan, lhs, rhs);

    /// <summary>
    /// call select function.
    /// </summary>
    /// <param name="predicate">conditon value.</param>
    /// <param name="true_value">lhs.</param>
    /// <param name="false_value">rhs.</param>
    public static Call Select(Expr predicate, Expr true_value, Expr false_value) => new Call(new Select(), predicate, true_value, false_value);

    /// <summary>
    /// call condition.
    /// </summary>
    public static Call Condition(Expr predicate, Expr value) => new Call(new Condition(), predicate, value);

    /// <summary>
    /// call select function.
    /// </summary>
    /// <param name="predicate">conditon value.</param>
    /// <param name="value">value.</param>
    /// <param name="message">requrie message.</param>
    public static Call Require(Expr predicate, Expr value, [System.Runtime.CompilerServices.CallerArgumentExpression("predicate")] string? message = null) => new Call(new Require(message!), predicate, value);

    public static Call RangeOf(Expr input)
    {
        var call = (Call)new Call(new RangeOf(), input).InheritMetaData(input);

        return call;
    }

    public static Call QuantParamOf(QuantMode mode, Expr range, Expr bits) => new Call(new QuantParamOf(mode), range, bits);

    public static Call Quantize(Expr input, Expr quantParam, DataType targetType) => new Call(new Quantize(targetType), input, quantParam);

    public static Call Dequantize(Expr input, Expr quantParam, DataType targetType) => new Call(new Dequantize(targetType), input, quantParam);

    public static Call FakeQuantize(Expr input, Expr quantParam, DataType targetType) => new Call(new FakeQuantize(targetType), input, quantParam);

    public static Call FakeDequantize(Expr input, Expr quantParam, DataType targetType) => new Call(new FakeDequantize(targetType), input, quantParam);

    /// <summary>
    /// attach the rangeof on the target, when run the egraph pass, will replace the rangeof expression with the constant.
    /// </summary>
    /// <returns> new marker expression. </returns>
    public static Marker RangeOfMarker(Expr target, Expr range)
    {
        var call = (Marker)new Marker(WellknownMarkerNames.RangeOf, target, range).InheritMetaData(target);

        return call;
    }

    public static Marker RangeOfMarker(Expr target, Expr range, DataType markerQuantType)
    {
        var call = (Marker)new Marker(WellknownMarkerNames.RangeOf, target, range).InheritMetaData(target);
        if (call.MixQuantInfo == null)
        {
            call.MixQuantInfo = new MixQuantInfo();
        }

        call.MixQuantInfo!.MarkerQuantType = markerQuantType;

        return call;
    }
}
