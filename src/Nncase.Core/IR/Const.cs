// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using NetFabric.Hyperlinq;

namespace Nncase.IR;

/// <summary>
/// Constant expression.
/// </summary>
public abstract class Const : Expr
{
    /// <summary>
    /// Initializes a new instance of the <see cref="Const"/> class.
    /// </summary>
    /// <param name="valueType">Type of value.</param>
    public Const(IRType valueType)
        : base(Array.Empty<Expr>())
    {
        ValueType = valueType;
    }

    public IRType ValueType { get; }

    /// <summary>
    /// Create constant from a <see cref="byte"/>.
    /// </summary>
    /// <param name="value">Value.</param>
    public static implicit operator Const(byte value) => FromTensor(Tensor.FromScalar(value));

    /// <summary>
    /// Create constant from a <see cref="ushort"/>.
    /// </summary>
    /// <param name="value">Value.</param>
    public static implicit operator Const(ushort value) => FromTensor(Tensor.FromScalar(value));

    /// <summary>
    /// Create constant from a <see cref="uint"/>.
    /// </summary>
    /// <param name="value">Value.</param>
    public static implicit operator Const(uint value) => FromTensor(Tensor.FromScalar(value));

    /// <summary>
    /// Create constant from a <see cref="ulong"/>.
    /// </summary>
    /// <param name="value">Value.</param>
    public static implicit operator Const(ulong value) => FromTensor(Tensor.FromScalar(value));

    /// <summary>
    /// Create constant from a <see cref="sbyte"/>.
    /// </summary>
    /// <param name="value">Value.</param>
    public static implicit operator Const(sbyte value) => FromTensor(Tensor.FromScalar(value));

    /// <summary>
    /// Create constant from a <see cref="short"/>.
    /// </summary>
    /// <param name="value">Value.</param>
    public static implicit operator Const(short value) => FromTensor(Tensor.FromScalar(value));

    /// <summary>
    /// Create constant from a <see cref="int"/>.
    /// </summary>
    /// <param name="value">Value.</param>
    public static implicit operator Const(int value) => FromTensor(Tensor.FromScalar(value));

    /// <summary>
    /// Create constant from a <see cref="long"/>.
    /// </summary>
    /// <param name="value">Value.</param>
    public static implicit operator Const(long value) => FromTensor(Tensor.FromScalar(value));

    /// <summary>
    /// Create constant from a <see cref="Half"/>.
    /// </summary>
    /// <param name="value">Value.</param>
    public static implicit operator Const(Half value) => FromTensor(Tensor.FromScalar(value));

    /// <summary>
    /// Create constant from a <see cref="float"/>.
    /// </summary>
    /// <param name="value">Value.</param>
    public static implicit operator Const(float value) => FromTensor(Tensor.FromScalar(value));

    /// <summary>
    /// Create constant from a <see cref="double"/>.
    /// </summary>
    /// <param name="value">Value.</param>
    public static implicit operator Const(double value) => FromTensor(Tensor.FromScalar(value));

    /// <summary>
    /// Create constant from a <see cref="BFloat16"/>.
    /// </summary>
    /// <param name="value">Value.</param>
    public static implicit operator Const(BFloat16 value) => FromTensor(Tensor.FromScalar(value));

    /// <summary>
    /// Create constant from a <see cref="bool"/>.
    /// </summary>
    /// <param name="value">Value.</param>
    public static implicit operator Const(bool value) => FromTensor(Tensor.FromScalar(value));

    /// <summary>
    /// Create constant from a <see cref="Utf8Char"/>.
    /// </summary>
    /// <param name="value">Value.</param>
    public static implicit operator Const(Utf8Char value) => FromTensor(Tensor.FromScalar(value));

    /// <summary>
    /// Create constant from <see cref="string"/>.
    /// </summary>
    /// <param name="value">Value.</param>
    public static implicit operator Const(string value) => FromTensor(Tensor.From<char>(value.ToCharArray()));

    /// <summary>
    /// Create constant from a tensor.
    /// </summary>
    /// <param name="tensor">Tensor.</param>
    /// <returns>Created constant expression.</returns>
    public static TensorConst FromTensor(Tensor tensor)
      => new(tensor);

    /// <summary>
    /// convert shape to const expr.
    /// </summary>
    public static Const FromShape(Shape shape) => FromTensor(Tensor.From<int>(shape.ToValueArray()));

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
            return new TupleConst(tpv);
        }
    }
}
