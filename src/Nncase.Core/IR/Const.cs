// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics.Tensors;
using NetFabric.Hyperlinq;

namespace Nncase.IR;

/// <summary>
/// Constant expression.
/// </summary>
public abstract record Const(IRType ValueType) : Expr
{
    /// <summary>
    /// Create constant from a <see cref="byte"/>.
    /// </summary>
    /// <param name="value">Value.</param>
    public static implicit operator Const(byte value) => FromScalar(value);

    /// <summary>
    /// Create constant from a <see cref="ushort"/>.
    /// </summary>
    /// <param name="value">Value.</param>
    public static implicit operator Const(ushort value) => FromScalar(value);

    /// <summary>
    /// Create constant from a <see cref="uint"/>.
    /// </summary>
    /// <param name="value">Value.</param>
    public static implicit operator Const(uint value) => FromScalar(value);

    /// <summary>
    /// Create constant from a <see cref="ulong"/>.
    /// </summary>
    /// <param name="value">Value.</param>
    public static implicit operator Const(ulong value) => FromScalar(value);

    /// <summary>
    /// Create constant from a <see cref="sbyte"/>.
    /// </summary>
    /// <param name="value">Value.</param>
    public static implicit operator Const(sbyte value) => FromScalar(value);

    /// <summary>
    /// Create constant from a <see cref="short"/>.
    /// </summary>
    /// <param name="value">Value.</param>
    public static implicit operator Const(short value) => FromScalar(value);

    /// <summary>
    /// Create constant from a <see cref="int"/>.
    /// </summary>
    /// <param name="value">Value.</param>
    public static implicit operator Const(int value) => FromScalar(value);

    /// <summary>
    /// Create constant from a <see cref="long"/>.
    /// </summary>
    /// <param name="value">Value.</param>
    public static implicit operator Const(long value) => FromScalar(value);

    /// <summary>
    /// Create constant from a <see cref="Half"/>.
    /// </summary>
    /// <param name="value">Value.</param>
    public static implicit operator Const(Half value) => FromScalar(value);

    /// <summary>
    /// Create constant from a <see cref="float"/>.
    /// </summary>
    /// <param name="value">Value.</param>
    public static implicit operator Const(float value) => FromScalar(value);

    /// <summary>
    /// Create constant from a <see cref="double"/>.
    /// </summary>
    /// <param name="value">Value.</param>
    public static implicit operator Const(double value) => FromScalar(value);

    /// <summary>
    /// Create constant from a <see cref="BFloat16"/>.
    /// </summary>
    /// <param name="value">Value.</param>
    public static implicit operator Const(BFloat16 value) => FromScalar(value);

    /// <summary>
    /// Create constant from a <see cref="bool"/>.
    /// </summary>
    /// <param name="value">Value.</param>
    public static implicit operator Const(bool value) => FromScalar(value);

    /// <summary>
    /// Create constant from <see cref="string"/>.
    /// </summary>
    /// <param name="value">Value.</param>
    public static implicit operator Const(string value) => FromSpan<char>(value);

    /// <summary>
    /// Create constant from a scalar.
    /// </summary>
    /// <typeparam name="T">CLR type.</typeparam>
    /// <param name="value">Value.</param>
    /// <returns>Created constant expression.</returns>
    public static TensorConst FromScalar<T>(T value)
        where T : unmanaged, IEquatable<T>
        => new(Tensor.FromScalar(value));

    /// <summary>
    /// Create constant from a span.
    /// </summary>
    /// <typeparam name="T">CLR type.</typeparam>
    /// <param name="span">Span.</param>
    /// <param name="dimensions">Dimensions.</param>
    /// <returns>Created constant expression.</returns>
    public static TensorConst FromSpan<T>(ReadOnlySpan<T> span, ReadOnlySpan<int> dimensions)
        where T : unmanaged, IEquatable<T>
        => new(Tensor.FromSpan(span, dimensions));

    /// <summary>
    /// Create constant from a span, Set the shape as [n].
    /// </summary>
    /// <typeparam name="T">CLR type.</typeparam>
    /// <param name="span">Span.</param>
    /// <returns>Created constant expression.</returns>
    public static TensorConst FromSpan<T>(ReadOnlySpan<T> span)
        where T : unmanaged, IEquatable<T>
        => new(Tensor.FromSpan(span));

    /// <summary>
    /// Create constant from a tensor.
    /// </summary>
    /// <param name="tensor">Tensor.</param>
    /// <returns>Created constant expression.</returns>
    public static TensorConst FromTensor(Tensor tensor)
      => new(tensor);

    /// <summary>
    /// Create Constant Pointer
    /// </summary>
    /// <param name="addr"> the addr value.</param>
    /// <param name="code">pointed element type code.</param>
    /// <returns></returns>
    public static TensorConst FromPointer(long addr, PrimTypeCode code = PrimTypeCode.Float32) => new(Tensor.FromBytes(TensorType.Pointer(code), DataTypes.GetBytes(addr)));

    /// <summary>
    /// convert shape to const expr.
    /// </summary>
    /// <param name="shape"></param>
    /// <returns></returns>
    public static Const FromShape(Shape shape) => FromSpan<int>(shape.ToValueArray());

    /// <summary>
    /// Convert value to const expr.
    /// </summary>
    /// <param name="value">Value.</param>
    /// <returns>Created constant expression.</returns>
    public static Const FromValue(IValue value)
    {
        if (value is TensorValue tv)
        {
            return new TensorConst(tv.AsTensor());
        }
        else
        {
            var tpv = (TupleValue)value;
            return new TupleConst(tpv.Select(x => FromValue(x)).ToArray());
        }
    }

}

