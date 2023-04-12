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
    /// Create Tensor from a memory of <see cref="byte"/>.
    /// </summary>
    /// <param name="value">Value.</param>
    public static implicit operator Tensor(Memory<byte> value) => From(value);

    /// <summary>
    /// Create Tensor from a memory of <see cref="ushort"/>.
    /// </summary>
    /// <param name="value">Value.</param>
    public static implicit operator Tensor(Memory<ushort> value) => From(value);

    /// <summary>
    /// Create Tensor from a memory of <see cref="uint"/>.
    /// </summary>
    /// <param name="value">Value.</param>
    public static implicit operator Tensor(Memory<uint> value) => From(value);

    /// <summary>
    /// Create Tensor from a memory of <see cref="ulong"/>.
    /// </summary>
    /// <param name="value">Value.</param>
    public static implicit operator Tensor(Memory<ulong> value) => From(value);

    /// <summary>
    /// Create Tensor from a memory of <see cref="sbyte"/>.
    /// </summary>
    /// <param name="value">Value.</param>
    public static implicit operator Tensor(Memory<sbyte> value) => From(value);

    /// <summary>
    /// Create Tensor from a memory of <see cref="short"/>.
    /// </summary>
    /// <param name="value">Value.</param>
    public static implicit operator Tensor(Memory<short> value) => From(value);

    /// <summary>
    /// Create Tensor from a memory of <see cref="int"/>.
    /// </summary>
    /// <param name="value">Value.</param>
    public static implicit operator Tensor(Memory<int> value) => From(value);

    /// <summary>
    /// Create Tensor from a memory of <see cref="long"/>.
    /// </summary>
    /// <param name="value">Value.</param>
    public static implicit operator Tensor(Memory<long> value) => From(value);

    /// <summary>
    /// Create Tensor from a memory of <see cref="Half"/>.
    /// </summary>
    /// <param name="value">Value.</param>
    public static implicit operator Tensor(Memory<Half> value) => From(value);

    /// <summary>
    /// Create Tensor from a memory of <see cref="float"/>.
    /// </summary>
    /// <param name="value">Value.</param>
    public static implicit operator Tensor(Memory<float> value) => From(value);

    /// <summary>
    /// Create Tensor from a memory of <see cref="double"/>.
    /// </summary>
    /// <param name="value">Value.</param>
    public static implicit operator Tensor(Memory<double> value) => From(value);

    /// <summary>
    /// Create Tensor from a memory of <see cref="BFloat16"/>.
    /// </summary>
    /// <param name="value">Value.</param>
    public static implicit operator Tensor(Memory<BFloat16> value) => From(value);

    /// <summary>
    /// Create Tensor from a memory of <see cref="bool"/>.
    /// </summary>
    /// <param name="value">Value.</param>
    public static implicit operator Tensor(Memory<bool> value) => From(value);

    /// <summary>
    /// Create Tensor from an <see cref="Array"/>.
    /// </summary>
    /// <param name="value">Value.</param>
    public static implicit operator Tensor(Array value) => FromArray(value);

    /// <summary>
    /// Create value from a <see cref="Tensor"/>.
    /// </summary>
    /// <param name="tensor">Tensor.</param>
    public static implicit operator TensorValue(Tensor tensor) => Value.FromTensor(tensor);
}
