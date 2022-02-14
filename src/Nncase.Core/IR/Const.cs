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
    /// specific cast to float
    /// </summary>
    /// <returns></returns>
    public DenseTensor<float> HalfToFloat()
    {
        var src = (byte[])Data;
        var src_stride = DataTypes.GetLength(ValueType.DType);
        var n = src.Length / src_stride;
        var dest = new float[n];
        for (int i = 0; i < n; i++)
        {
            dest[i] = (float)DataTypes.CastToScalar(src, src_stride * i);
        }
    }
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
    /// cast to target type dense tensor
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <returns> tensor </returns>
    public DenseTensor<T> ToTensor<T>()
       where T : unmanaged
    {
        var srcType = ValueType.DType;
        var destType = DataTypes.FromType<T>();
        if (srcType == destType)
            return new DenseTensor<T>(Data.ToMemory<T>(), ValueType.IsScalar ? new[] { 1 } : ValueType.Shape);
        else
            return CastToTensor<T>(srcType);
    }

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
    /// convert const to scalar
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <returns></returns>
    /// <exception cref="InvalidCastException"></exception>
    /// <exception cref="NotSupportedException"></exception>
    public T ToScalar<T>()
      where T : unmanaged
    {
        if (!ValueType.IsScalar)
            throw new InvalidCastException($"This Const is Not Scalar!");
        return ValueType.DType switch
        {
            PrimType ptype => ptype.Lanes == 1 ?
          (DataTypes.FromType<T>() == ptype ?
               DataTypes.ToScalar<T>(ptype, Data, 0) :
               DataTypes.CastToScalar<T>(ptype, Data, 0)) : throw new InvalidCastException($"This Const Datatype Is Packed!"),
            PointerType potype => typeof(T) == typeof(long) ? DataTypes.ToScalar<T>(DataType.Int64, Data, 0) : throw new InvalidCastException($"The Const PointerType Only Can Convert To Int64!"),
            _ => throw new NotSupportedException(ValueType.DType.GetType().Name),
        };
    }

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

    /// <summary>
    /// convert shape to const expr.
    /// </summary>
    /// <param name="shape"></param>
    /// <returns></returns>
    public static Const FromShape(Shape shape) => FromSpan<int>(shape.Select(x => x.FixedValue).ToArray(), new[] { shape.Rank });


    /// <summary>
    /// convert const to string.
    /// </summary>
    /// <returns></returns>
    public override string ToString() => ValueType switch
    {
        TensorType tensorType => tensorType switch
        {
            var x when x.IsScalar =>
              x.DType switch
              {
                  var dtype when DataTypes.IsIntegral(dtype) => ToScalar<long>().ToString(),
                  var dtype when DataTypes.IsFloat(dtype) => ToScalar<float>().ToString(),
                  var dtype when DataTypes.IsPointer(dtype) => ToScalar<long>().ToString(),
                  _ => $"{x.DType.ToString()} {x.Shape}"
              },
            _ => $"{tensorType.DType.ToString()} {tensorType.Shape}"
        },
        TupleType tupleType => tupleType.ToString(),
        var x => throw new NotSupportedException($"When ValueType Is {x.GetType().Name}")
    };
}

