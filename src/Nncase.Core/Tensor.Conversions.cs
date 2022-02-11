// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase;

public partial class Tensor
{
    /// <summary>
    /// Create Tensor from a <see cref="byte"/>.
    /// </summary>
    /// <param name="value">Value.</param>
    public static implicit operator Tensor(byte value) => FromScalar(value);

    /// <summary>
    /// Create Tensor from a <see cref="ushort"/>.
    /// </summary>
    /// <param name="value">Value.</param>
    public static implicit operator Tensor(ushort value) => FromScalar(value);

    /// <summary>
    /// Create Tensor from a <see cref="uint"/>.
    /// </summary>
    /// <param name="value">Value.</param>
    public static implicit operator Tensor(uint value) => FromScalar(value);

    /// <summary>
    /// Create Tensor from a <see cref="ulong"/>.
    /// </summary>
    /// <param name="value">Value.</param>
    public static implicit operator Tensor(ulong value) => FromScalar(value);

    /// <summary>
    /// Create Tensor from a <see cref="sbyte"/>.
    /// </summary>
    /// <param name="value">Value.</param>
    public static implicit operator Tensor(sbyte value) => FromScalar(value);

    /// <summary>
    /// Create Tensor from a <see cref="short"/>.
    /// </summary>
    /// <param name="value">Value.</param>
    public static implicit operator Tensor(short value) => FromScalar(value);

    /// <summary>
    /// Create Tensor from a <see cref="int"/>.
    /// </summary>
    /// <param name="value">Value.</param>
    public static implicit operator Tensor(int value) => FromScalar(value);

    /// <summary>
    /// Create Tensor from a <see cref="long"/>.
    /// </summary>
    /// <param name="value">Value.</param>
    public static implicit operator Tensor(long value) => FromScalar(value);

    /// <summary>
    /// Create Tensor from a <see cref="Half"/>.
    /// </summary>
    /// <param name="value">Value.</param>
    public static implicit operator Tensor(Half value) => FromScalar(value);

    /// <summary>
    /// Create Tensor from a <see cref="float"/>.
    /// </summary>
    /// <param name="value">Value.</param>
    public static implicit operator Tensor(float value) => FromScalar(value);

    /// <summary>
    /// Create Tensor from a <see cref="double"/>.
    /// </summary>
    /// <param name="value">Value.</param>
    public static implicit operator Tensor(double value) => FromScalar(value);

    /// <summary>
    /// Create Tensor from a <see cref="BFloat16"/>.
    /// </summary>
    /// <param name="value">Value.</param>
    public static implicit operator Tensor(BFloat16 value) => FromScalar(value);

    /// <summary>
    /// Create Tensor from a <see cref="bool"/>.
    /// </summary>
    /// <param name="value">Value.</param>
    public static implicit operator Tensor(bool value) => FromScalar(value);

    /// <summary>
    /// Create Tensor from a span of <see cref="byte"/>.
    /// </summary>
    /// <param name="value">Value.</param>
    public static implicit operator Tensor(ReadOnlySpan<byte> value) => FromSpan(value);

    /// <summary>
    /// Create Tensor from a span of <see cref="ushort"/>.
    /// </summary>
    /// <param name="value">Value.</param>
    public static implicit operator Tensor(ReadOnlySpan<ushort> value) => FromSpan(value);

    /// <summary>
    /// Create Tensor from a span of <see cref="uint"/>.
    /// </summary>
    /// <param name="value">Value.</param>
    public static implicit operator Tensor(ReadOnlySpan<uint> value) => FromSpan(value);

    /// <summary>
    /// Create Tensor from a span of <see cref="ulong"/>.
    /// </summary>
    /// <param name="value">Value.</param>
    public static implicit operator Tensor(ReadOnlySpan<ulong> value) => FromSpan(value);

    /// <summary>
    /// Create Tensor from a span of <see cref="sbyte"/>.
    /// </summary>
    /// <param name="value">Value.</param>
    public static implicit operator Tensor(ReadOnlySpan<sbyte> value) => FromSpan(value);

    /// <summary>
    /// Create Tensor from a span of <see cref="short"/>.
    /// </summary>
    /// <param name="value">Value.</param>
    public static implicit operator Tensor(ReadOnlySpan<short> value) => FromSpan(value);

    /// <summary>
    /// Create Tensor from a span of <see cref="int"/>.
    /// </summary>
    /// <param name="value">Value.</param>
    public static implicit operator Tensor(ReadOnlySpan<int> value) => FromSpan(value);

    /// <summary>
    /// Create Tensor from a span of <see cref="long"/>.
    /// </summary>
    /// <param name="value">Value.</param>
    public static implicit operator Tensor(ReadOnlySpan<long> value) => FromSpan(value);

    /// <summary>
    /// Create Tensor from a span of <see cref="Half"/>.
    /// </summary>
    /// <param name="value">Value.</param>
    public static implicit operator Tensor(ReadOnlySpan<Half> value) => FromSpan(value);

    /// <summary>
    /// Create Tensor from a span of <see cref="float"/>.
    /// </summary>
    /// <param name="value">Value.</param>
    public static implicit operator Tensor(ReadOnlySpan<float> value) => FromSpan(value);

    /// <summary>
    /// Create Tensor from a span of <see cref="double"/>.
    /// </summary>
    /// <param name="value">Value.</param>
    public static implicit operator Tensor(ReadOnlySpan<double> value) => FromSpan(value);

    /// <summary>
    /// Create Tensor from a span of <see cref="BFloat16"/>.
    /// </summary>
    /// <param name="value">Value.</param>
    public static implicit operator Tensor(ReadOnlySpan<BFloat16> value) => FromSpan(value);

    /// <summary>
    /// Create Tensor from a span of <see cref="bool"/>.
    /// </summary>
    /// <param name="value">Value.</param>
    public static implicit operator Tensor(ReadOnlySpan<bool> value) => FromSpan(value);

    /// <summary>
    /// Create value from a <see cref="Tensor"/>.
    /// </summary>
    /// <param name="tensor">Tensor.</param>
    public static implicit operator TensorValue(Tensor tensor) => Value.FromTensor(tensor);
}
