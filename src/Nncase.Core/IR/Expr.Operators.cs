﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NetFabric.Hyperlinq;
using Nncase.IR.Tensors;
using static Nncase.IR.F.Math;

namespace Nncase.IR;

/// <summary>
/// Math operators for <see cref="Expr"/>.
/// </summary>
public partial class Expr
{
    /// <summary>
    /// get the item from the expr.
    /// </summary>
    /// <param name="index"> expr. </param>
    public Expr this[Expr index] => F.Tensors.GetItem(this, index);

    /// <summary>
    /// get the item from the expr.
    /// </summary>
    /// <returns> expr. </returns>
    public Expr this[long index] =>
        this switch
        {
            TensorConst tc when tc.Value.Rank == 1 => Tensor.FromScalar(tc.Value.ElementType, tc.Value[index]),
            TupleConst tc => tc.Value[(int)index].AsTensor(),
            Shape shape => shape.Dimensions[(int)index],
            IR.Tuple t => t[(int)index],
            Call { Target: Concat { Axis: 0 } } c when c[Concat.Input] is IR.Tuple tp && tp.Fields[0].CheckedType is TensorType { Shape: { IsFixed: true, Size: 1 } } => c[Concat.Input][index][0],
            Call { Target: Reshape } c when c[Reshape.Shape] is TensorConst tc && tc.Value.Length == 1 && tc.Value.Cast<long>()[0] == 1 => c[Reshape.Input],
            Call { Target: Stack } c => c[Stack.Inputs][index],
            _ => this[(Expr)index],
        };

    /// <summary>
    /// get the item from the expr.
    /// </summary>
    /// <returns> expr. </returns>
    public Expr this[params long[] indices] =>
        this switch
        {
            TensorConst tc when tc.Value.Rank == indices.Length => Tensor.FromScalar(tc.Value.ElementType, tc.Value[indices]),
            Call { Target: Stack } c when indices.Length == 2 => c[Stack.Inputs][indices[0]][indices[1]],
            _ => this[indices.Select(x => (Expr)x).ToArray()],
        };

    /// <summary>
    /// get the item from the expr.
    /// </summary>
    /// <returns> expr. </returns>
    public Expr this[params Expr[] indices]
    {
        get
        {
            if (indices.Length == 0)
            {
                return this;
            }
            else
            {
                return F.Tensors.GetItem(this, F.Tensors.Stack(new IR.Tuple(indices), 0));
            }
        }
    }

    public Expr this[int index] => this[(long)index];

    public Expr this[Index index]
    {
        get
        {
            if (index.IsFromEnd)
            {
                var shape = (CheckedType as TensorType)?.Shape;
                if (shape?.IsRanked == true)
                {
                    var newIndex = shape[0] - index.Value;
                    return newIndex.IsFixed ? this[newIndex.FixedValue] : this[newIndex.ToExpr()];
                }
                else
                {
                    return this[IR.F.Tensors.ShapeOf(this)[0] - (long)index.Value];
                }
            }

            return this[(long)index.Value];
        }
    }

    /// <summary>
    /// Unary neg.
    /// </summary>
    /// <param name="lhs">Source operand.</param>
    /// <returns>Result.</returns>
    public static Call operator -(Expr lhs) => Neg(lhs);

    /// <summary>
    /// Unary bitwise not.
    /// </summary>
    /// <param name="lhs">Source operand.</param>
    /// <returns>Result.</returns>
    public static Call operator ~(Expr lhs) => BitwiseNot(lhs);

    /// <summary>
    /// Unary logical not.
    /// </summary>
    /// <param name="lhs">Source operand.</param>
    /// <returns>Result.</returns>
    public static Call operator !(Expr lhs) => LogicalNot(lhs);

    /// <summary>
    /// Binary add.
    /// </summary>
    /// <param name="lhs">Left operand.</param>
    /// <param name="rhs">Right operand.</param>
    /// <returns>Result.</returns>
    public static Call operator +(Expr lhs, Expr rhs) => Add(lhs, rhs);

    /// <summary>
    /// Binary sub.
    /// </summary>
    /// <param name="lhs">Left operand.</param>
    /// <param name="rhs">Right operand.</param>
    /// <returns>Result.</returns>
    public static Call operator -(Expr lhs, Expr rhs) => Sub(lhs, rhs);

    /// <summary>
    /// Binary mul.
    /// </summary>
    /// <param name="lhs">Left operand.</param>
    /// <param name="rhs">Right operand.</param>
    /// <returns>Result.</returns>
    public static Call operator *(Expr lhs, Expr rhs) => Mul(lhs, rhs);

    /// <summary>
    /// Binary div.
    /// </summary>
    /// <param name="lhs">Left operand.</param>
    /// <param name="rhs">Right operand.</param>
    /// <returns>Result.</returns>
    public static Call operator /(Expr lhs, Expr rhs) => Div(lhs, rhs);

    /// <summary>
    /// Binary mod.
    /// </summary>
    /// <param name="lhs">Left operand.</param>
    /// <param name="rhs">Right operand.</param>
    /// <returns>Result.</returns>
    public static Call operator %(Expr lhs, Expr rhs) => Mod(lhs, rhs);

    /// <summary>
    /// Binary bitwise and.
    /// </summary>
    /// <param name="lhs">Left operand.</param>
    /// <param name="rhs">Right operand.</param>
    /// <returns>Result.</returns>
    public static Call operator &(Expr lhs, Expr rhs) => BitwiseAnd(lhs, rhs);

    /// <summary>
    /// Binary bitwise or.
    /// </summary>
    /// <param name="lhs">Left operand.</param>
    /// <param name="rhs">Right operand.</param>
    /// <returns>Result.</returns>
    public static Call operator |(Expr lhs, Expr rhs) => BitwiseOr(lhs, rhs);

    /// <summary>
    /// Binary bitwise xor.
    /// </summary>
    /// <param name="lhs">Left operand.</param>
    /// <param name="rhs">Right operand.</param>
    /// <returns>Result.</returns>
    public static Call operator ^(Expr lhs, Expr rhs) => BitwiseXor(lhs, rhs);

    /// <summary>
    /// Binary left shift.
    /// </summary>
    /// <param name="lhs">Left operand.</param>
    /// <param name="rhs">Right operand.</param>
    /// <returns>Result.</returns>
    public static Call operator <<(Expr lhs, int rhs) => LeftShift(lhs, rhs);

    /// <summary>
    /// Binary right shift.
    /// </summary>
    /// <param name="lhs">Left operand.</param>
    /// <param name="rhs">Right operand.</param>
    /// <returns>Result.</returns>
    public static Call operator >>(Expr lhs, int rhs) => LeftShift(lhs, rhs);

    /// <summary>
    /// GreaterEqual.
    /// </summary>
    public static Call operator >=(Expr lhs, Expr rhs) => GreaterEqual(lhs, rhs);

    /// <summary>
    /// GreaterThan.
    /// </summary>
    public static Call operator >(Expr lhs, Expr rhs) => GreaterThan(lhs, rhs);

    /// <summary>
    /// LessEqual.
    /// </summary>
    public static Call operator <=(Expr lhs, Expr rhs) => LessEqual(lhs, rhs);

    /// <summary>
    /// LessThan.
    /// </summary>
    public static Call operator <(Expr lhs, Expr rhs) => LessThan(lhs, rhs);
}
