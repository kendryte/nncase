// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.IR;

/// <summary>
/// Constant of tensor.
/// </summary>
public sealed record TensorConst(Tensor Value) : Const(new TensorType(Value.ElementType, Value.Shape))
{
    /// <summary>
    /// Gets value type.
    /// </summary>
    public new TensorType ValueType => (TensorType)base.ValueType;

    /// <summary>
    /// Create TensorConstant from a <see cref="byte"/>.
    /// </summary>
    /// <param name="value">Value.</param>
    public static implicit operator TensorConst(byte value) => new(Tensor.FromScalar(value));

    /// <summary>
    /// Create TensorConstant from a <see cref="ushort"/>.
    /// </summary>
    /// <param name="value">Value.</param>
    public static implicit operator TensorConst(ushort value) => new(Tensor.FromScalar(value));

    /// <summary>
    /// Create TensorConstant from a <see cref="uint"/>.
    /// </summary>
    /// <param name="value">Value.</param>
    public static implicit operator TensorConst(uint value) => new(Tensor.FromScalar(value));

    /// <summary>
    /// Create TensorConstant from a <see cref="ulong"/>.
    /// </summary>
    /// <param name="value">Value.</param>
    public static implicit operator TensorConst(ulong value) => new(Tensor.FromScalar(value));

    /// <summary>
    /// Create TensorConstant from a <see cref="sbyte"/>.
    /// </summary>
    /// <param name="value">Value.</param>
    public static implicit operator TensorConst(sbyte value) => new(Tensor.FromScalar(value));

    /// <summary>
    /// Create TensorConstant from a <see cref="short"/>.
    /// </summary>
    /// <param name="value">Value.</param>
    public static implicit operator TensorConst(short value) => new(Tensor.FromScalar(value));

    /// <summary>
    /// Create TensorConstant from a <see cref="int"/>.
    /// </summary>
    /// <param name="value">Value.</param>
    public static implicit operator TensorConst(int value) => new(Tensor.FromScalar(value));

    /// <summary>
    /// Create TensorConstant from a <see cref="long"/>.
    /// </summary>
    /// <param name="value">Value.</param>
    public static implicit operator TensorConst(long value) => new(Tensor.FromScalar(value));

    /// <summary>
    /// Create TensorConstant from a <see cref="Half"/>.
    /// </summary>
    /// <param name="value">Value.</param>
    public static implicit operator TensorConst(Half value) => new(Tensor.FromScalar(value));

    /// <summary>
    /// Create TensorConstant from a <see cref="float"/>.
    /// </summary>
    /// <param name="value">Value.</param>
    public static implicit operator TensorConst(float value) => new(Tensor.FromScalar(value));

    /// <summary>
    /// Create TensorConstant from a <see cref="double"/>.
    /// </summary>
    /// <param name="value">Value.</param>
    public static implicit operator TensorConst(double value) => new(Tensor.FromScalar(value));

    /// <summary>
    /// Create TensorConstant from a <see cref="BFloat16"/>.
    /// </summary>
    /// <param name="value">Value.</param>
    public static implicit operator TensorConst(BFloat16 value) => new(Tensor.FromScalar(value));

    /// <summary>
    /// Create TensorConstant from a <see cref="bool"/>.
    /// </summary>
    /// <param name="value">Value.</param>
    public static implicit operator TensorConst(bool value) => new(Tensor.FromScalar(value));

    /// <summary>
    /// Create TensorConstant from <see cref="string"/>.
    /// </summary>
    /// <param name="value">Value.</param>
    public static implicit operator TensorConst(string value) => new(Tensor.From<char>(value.ToCharArray()));

    /// <inheritdoc/>
    public override string ToString() => ValueType switch
    {
        var x when x.IsScalar =>
          x.DType switch
          {
              var dtype when DataTypes.IsIntegral(dtype) => Value.ToScalar<long>().ToString(),
              var dtype when DataTypes.IsFloat(dtype) => Value.ToScalar<float>().ToString(),
              var dtype when DataTypes.IsPointer(dtype) => Value.ToScalar<ulong>().ToString(),
              var dtype when dtype == DataTypes.Boolean => Value.ToScalar<bool>().ToString(),
              _ => $"{x.DType.GetDisplayName()} {x.Shape}",
          },
        _ => $"{ValueType.DType.GetDisplayName()} {ValueType.Shape}",
    };

    /// <inheritdoc/>
    public override int GetHashCode()
    {
        return HashCodeCache ??= HashCode.Combine(
            EqualityComparer<Type>.Default.GetHashCode(EqualityContract),
            Value.GetHashCode());
    }
}
