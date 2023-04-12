// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.PatternMatch;

public abstract partial record Pattern
{
    public static implicit operator Pattern(byte value) => (TensorConst)value;

    public static implicit operator Pattern(ushort value) => (TensorConst)value;

    public static implicit operator Pattern(uint value) => (TensorConst)value;

    public static implicit operator Pattern(ulong value) => (TensorConst)value;

    public static implicit operator Pattern(sbyte value) => (TensorConst)value;

    public static implicit operator Pattern(short value) => (TensorConst)value;

    public static implicit operator Pattern(int value) => (TensorConst)value;

    public static implicit operator Pattern(long value) => (TensorConst)value;

    public static implicit operator Pattern(Half value) => (TensorConst)value;

    public static implicit operator Pattern(float value) => (TensorConst)value;

    public static implicit operator Pattern(double value) => (TensorConst)value;

    public static implicit operator Pattern(BFloat16 value) => (TensorConst)value;

    public static implicit operator Pattern(bool value) => (TensorConst)value;

    public static implicit operator Pattern(Tensor value) => (TensorConst)value;

    public static implicit operator Pattern(int[] span) => Const.FromTensor(Tensor.From<int>(span));

    public static implicit operator Pattern(float[] span) =>
        Const.FromTensor(Tensor.From<float>(span));

    /// <summary>
    /// Convert <see cref="TensorConst"/> to <see cref="Pattern"/>.
    /// </summary>
    /// <param name="con">Tensor const.</param>
    public static implicit operator Pattern(TensorConst con) => new TensorConstPattern(con, null);

    /// <summary>
    /// Convert <see cref="Expr"/> to <see cref="Pattern"/>.
    /// </summary>
    /// <param name="expr">Expression.</param>
    public static implicit operator Pattern(Expr expr) => expr switch
    {
        Var var => new VarPattern(var),
        TensorConst con => new TensorConstPattern(con, null),
        Const con => new ConstPattern(con, null),
        Call call => new CallPattern(call, null),
        IR.Tuple tuple => new TuplePattern(tuple, null),
        Op op => (Pattern)Activator.CreateInstance(typeof(OpPattern<>).MakeGenericType(op.GetType()), op)!,
        _ => throw new NotImplementedException($"Can't Convert The Expr {expr.GetType().Name} To Pattern"),
    };
}
