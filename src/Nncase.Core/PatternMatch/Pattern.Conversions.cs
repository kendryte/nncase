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
    public static implicit operator Pattern(byte value) => (TensorConstPattern)value;

    public static implicit operator Pattern(ushort value) => (TensorConstPattern)value;

    public static implicit operator Pattern(uint value) => (TensorConstPattern)value;

    public static implicit operator Pattern(ulong value) => (TensorConstPattern)value;

    public static implicit operator Pattern(sbyte value) => (TensorConstPattern)value;

    public static implicit operator Pattern(short value) => (TensorConstPattern)value;

    public static implicit operator Pattern(int value) => (TensorConstPattern)value;

    public static implicit operator Pattern(long value) => (TensorConstPattern)value;

    public static implicit operator Pattern(Half value) => (TensorConstPattern)value;

    public static implicit operator Pattern(float value) => (TensorConstPattern)value;

    public static implicit operator Pattern(double value) => (TensorConstPattern)value;

    public static implicit operator Pattern(BFloat16 value) => (TensorConstPattern)value;

    public static implicit operator Pattern(bool value) => (TensorConstPattern)value;

    public static implicit operator Pattern(Tensor value) => (TensorConstPattern)value;

    public static implicit operator Pattern(int[] span) => Const.FromTensor(Tensor.FromSpan<int>(span));

    public static implicit operator Pattern(float[] span) =>
        Const.FromTensor(Tensor.FromSpan<float>(span));

    /// <summary>
    /// Convert <see cref="Expr"/> to <see cref="Pattern"/>.
    /// </summary>
    /// <param name="expr">Expression.</param>
    public static implicit operator Pattern(Expr expr) => expr switch
    {
        Var var => new VarPattern(var),
        TensorConst con => new TensorConstPattern(con, null),
        Const con => new ConstPattern(con, null),
        Function function => new FunctionPattern(function, null),
        Call call => new CallPattern(call, null),
        IR.Tuple tuple => new TuplePattern(tuple, null),
        Op op => (Pattern)Activator.CreateInstance(typeof(OpPattern<>).MakeGenericType(op.GetType()), op)!,
        _ => throw new NotImplementedException($"Can't Convert The Expr {expr.GetType().Name} To Pattern"),
    };
}
