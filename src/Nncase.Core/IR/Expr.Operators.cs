// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
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
    public Expr this[params long[] indices] =>
        this switch
        {
            TensorConst tc => Tensor.FromScalar(tc.Value.ElementType, tc.Value[indices]),
            TupleConst tc => tc.Value[(int)indices.Single()].AsTensor(),
            Shape shape => shape.Dimensions[(int)indices.Single()],
            IR.Tuple t => t[(int)indices.Single()],
            Call { Target: Concat { Axis: 0 } } c when indices.Length == 1 => c[Concat.Input][indices[0]][0],
            Call { Target: Reshape } c when c[Reshape.Shape] is TensorConst tc && tc.Value.Length == 1 && tc.Value.ToScalar<long>() == 1 => c[Reshape.Input],
            Call { Target: Stack } c when indices.Length == 1 => c[Stack.Inputs][indices[0]],
            Call { Target: Fusion } c when indices.Length == 1 => GetItemOfBaseFunction(c, indices[0]),
            _ => this[indices.Select(x => (Expr)x).ToArray()],
        };

    private Expr GetItemOfBaseFunction(Call c, long v)
    {
        if (c.CheckedType is TupleType tt && tt.Count == 3 && v == 3)
        {
            Console.WriteLine();
        }

        return c[(Expr)v];
    }

    /// <summary>
    /// get the item from the expr.
    /// </summary>
    /// <returns> expr. </returns>
    public Expr this[params Expr[] indices] => F.Tensors.GetItem(this, F.Tensors.Stack(new IR.Tuple(indices), 0));

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
