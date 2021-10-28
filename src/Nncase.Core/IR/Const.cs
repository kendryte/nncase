// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Numerics.Tensors;
using System.Collections.Generic;

namespace Nncase.IR
{
    /// <summary>
    /// Constant expression.
    /// </summary>
    public sealed record Const(TensorType ValueType, IRBytes Data) : Expr
    {

        public List<Dimension> Shape => new List<Dimension>(ValueType.Shape);


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

        public DenseTensor<T> ToTensor<T>()
           where T : unmanaged
           => ValueType.DType == DataTypes.FromType<T>() ?
             new DenseTensor<T>(Data.ToMemory<T>(), ValueType.Shape) :
              throw new InvalidCastException($"The Target Type {DataTypes.FromType<T>().ToString()} Is Not Equal Current Type {ValueType.DType.ToString()}!");

        public T ToScalar<T>()
          where T : unmanaged
          => ValueType.IsScalar ?
           ToTensor<T>()[0] :
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

        public static Const FromTensor<T>(DenseTensor<T> ts)
          where T : unmanaged
          => FromSpan<T>(ts.Buffer.Span, new Shape(ts.Dimensions));

    }
}
