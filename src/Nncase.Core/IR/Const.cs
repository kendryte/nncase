// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Numerics.Tensors;
using System.Collections.Generic;
using System.Linq;
using NetFabric.Hyperlinq;

namespace Nncase.IR
{
    /// <summary>
    /// Constant expression.
    /// </summary>
    public sealed record Const(TensorType ValueType, IRBytes Data) : Expr
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
        /// Create constant from <see cref="string"/>
        /// </summary>
        /// <param name="value"></param>
        public static implicit operator Const(string value) => FromSpan<char>(value);

        /// <summary>
        /// <see cref="ToTensor{T}"/>
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="srcType"></param>
        /// <returns></returns>
        private DenseTensor<T> CastToTensor<T>(DataType srcType)
          where T : unmanaged
        {
            var src = (byte[])Data;
            var src_stride = DataTypes.GetLength(ValueType.DType);
            var n = src.Length / src_stride;
            var dest = new T[n];
            for (int i = 0; i < n; i++)
            {
                dest[i] = DataTypes.CastToScalar<T>(srcType, src, src_stride * i);
            }
            return new DenseTensor<T>(dest, ValueType.IsScalar ? new[] { 1 } : ValueType.Shape);
        }

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
        /// convert target type scalar
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <returns> scalar </returns>
        /// <exception cref="InvalidCastException"></exception>
        public T ToScalar<T>()
          where T : unmanaged
          => ValueType.IsScalar && ValueType.DType.Lanes == 1 ?
            (DataTypes.FromType<T>() == ValueType.DType ?
                 DataTypes.ToScalar<T>(ValueType.DType, Data, 0) :
                 DataTypes.CastToScalar<T>(ValueType.DType, Data, 0)) :
          throw new InvalidCastException($"This Const is Not Scalar!");

        /// <summary>
        /// cast to string.
        /// </summary>
        /// <returns> string </returns>
        /// <exception cref="InvalidCastException"></exception>
        public string ToStr() => ValueType.DType switch
        {
            { ElemType: ElemType.String, Lanes: 1 } => System.Text.Encoding.Default.GetString(Data),
            _ => throw new InvalidCastException($"This Const is Not String!")
        };

        /// <summary>
        /// Create constant from a scalar.
        /// </summary>
        /// <typeparam name="T">CLR type.</typeparam>
        /// <param name="value">Value.</param>
        /// <param name="lanes"> lanes </param>
        /// <returns>Created constant expression.</returns>
        public static Const FromScalar<T>(T value, int lanes = 1)
            where T : unmanaged
            => new(TensorType.Scalar(DataTypes.FromType<T>() with { Lanes = lanes }), RepeatBytes(DataTypes.GetBytes(value), lanes));

        /// <summary>
        /// repeat bytes
        /// </summary>
        /// <param name="bytes"></param>
        /// <param name="lanes"></param>
        /// <returns></returns>
        private static byte[] RepeatBytes(byte[] bytes, int lanes)
        {
            if (lanes == 1) return bytes;
            var ret = new byte[lanes * bytes.Length];
            for (int i = 0; i < lanes; i++)
            {
                bytes.CopyTo(ret, i * bytes.Length);
            }
            return ret;
        }

        /// <summary>
        /// Create constant from a span.
        /// </summary>
        /// <typeparam name="T">CLR type.</typeparam>
        /// <param name="span">Span.</param>
        /// <param name="shape">Shape.</param>
        /// <returns>Created constant expression.</returns>
        public static Const FromSpan<T>(ReadOnlySpan<T> span, Shape shape)
            where T : unmanaged
            => new(new TensorType(DataTypes.FromType<T>(), shape), DataTypes.GetBytes(span));

        /// <summary>
        /// Create constant from a span, Set the shape as [n]
        /// </summary>
        /// <typeparam name="T">CLR type.</typeparam>
        /// <param name="span">Span.</param>
        /// <returns>Created constant expression.</returns>
        public static Const FromSpan<T>(ReadOnlySpan<T> span)
            where T : unmanaged
            => new(new TensorType(DataTypes.FromType<T>(), new int[] { span.Length }), DataTypes.GetBytes(span));

        public static Const FromTensor<T>(DenseTensor<T> ts)
          where T : unmanaged
          => FromSpan<T>(ts.Buffer.Span, new Shape(ts.Dimensions.ToArray()));

        public static Const FromTensor<T>(Tensor<T> ts)
          where T : unmanaged
          => FromTensor<T>(ts.ToDenseTensor());

        public static Const FromTensor(DenseTensor<int> ts)
          => FromSpan<int>(ts.Buffer.Span, new Shape(ts.Dimensions.ToArray()));

        /// <summary>
        /// convert shape to const expr
        /// </summary>
        /// <param name="shape"></param>
        /// <returns></returns>
        public static Const FromShape(Shape shape) => FromSpan<int>(shape.Select(x => x.FixedValue).ToArray(), new[] { shape.Rank });

        /// <summary>
        /// get the const expr with specific shape 
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="shape"></param>
        /// <param name="value"></param>
        /// <returns></returns>
        public static Const FromShape<T>(Shape shape, T value)
         where T : unmanaged => FromTensor<T>(new DenseTensor<T>(Enumerable.Repeat<T>(value, shape.Size).ToArray(), shape));

        public override string ToString()
        {
            string str = DataTypes.GetDisplayName(ValueType.DType);
            if (ValueType.IsScalar)
            {
                if (DataTypes.IsIntegral(ValueType.DType))
                {
                    str = ToScalar<int>().ToString();
                }
                else if (DataTypes.IsFloat(ValueType.DType))
                {
                    str = ToScalar<float>().ToString();
                }
            }
            else
            {
                str += $" {ValueType.Shape}";
            }
            return str;
        }
    }
}
