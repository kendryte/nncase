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

        public override Shape Shape => ValueType.Shape;

        public override int Rank => ValueType.Shape.Rank;


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


        private DenseTensor<T> ToTensor<T>(DataType destType)
          where T : unmanaged
        {
            var src = (byte[])Data;
            var src_stride = DataTypes.GetLength(ValueType.DType);
            var n = src.Length / src_stride;
            var dest = new T[n];
            for (int i = 0; i < n; i++)
            {
                dest[i] = DataTypes.ToScalar<T>(destType, src, src_stride * i);
            }
            return new DenseTensor<T>(dest, ValueType.Shape);
        }

        public DenseTensor<T> ToTensor<T>()
           where T : unmanaged
        {
            var srcType = ValueType.DType;
            var destType = DataTypes.FromType<T>();
            if (srcType == destType)
                return new DenseTensor<T>(Data.ToMemory<T>(), ValueType.Shape);
            else
                return ToTensor<T>(destType);
        }

        public T ToScalar<T>()
          where T : unmanaged
          => ValueType.IsScalar ? DataTypes.ToScalar<T>(DataTypes.FromType<T>(), Data, 0) :
          throw new InvalidCastException($"This Const is Not Scalar!");

        /// <summary>
        /// Create constant from a scalar.
        /// </summary>
        /// <typeparam name="T">CLR type.</typeparam>
        /// <param name="value">Value.</param>
        /// <returns>Created constant expression.</returns>
        public static Const FromScalar<T>(T value)
            where T : unmanaged
            => new(TensorType.Scalar(DataTypes.FromType<T>()), DataTypes.GetBytes(value));

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
        /// <param name="shape">Shape.</param>
        /// <returns>Created constant expression.</returns>
        public static Const FromSpan<T>(ReadOnlySpan<T> span)
            where T : unmanaged
            => new(new TensorType(DataTypes.FromType<T>(), new int[] { span.Length }), DataTypes.GetBytes(span));

        public static Const FromTensor<T>(DenseTensor<T> ts)
          where T : unmanaged
          => FromSpan<T>(ts.Buffer.Span, new Shape(ts.Dimensions));

        public static Const FromTensor(DenseTensor<int> ts)
          => FromSpan<int>(ts.Buffer.Span, new Shape(ts.Dimensions));

        public static Const FromShape(Shape shape) => FromSpan<int>(shape.Select(x => x.FixedValue).ToArray(), new[] { shape.Rank });
    }
}
