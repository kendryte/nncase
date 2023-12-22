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
public sealed class TensorConst : Const, IEquatable<TensorConst?>
{
    public TensorConst(Tensor tensor)
        : base(new TensorType(tensor.ElementType, tensor.Shape))
    {
        Value = tensor;
    }

    public TensorConst(Tensor tensor, IRArray<SBP> ndsbp, Placement placement)
        : base(new DistributedType(new TensorType(tensor.ElementType, tensor.Shape), ndsbp, placement))
    {
        Value = tensor;
    }

    public Tensor Value { get; }

    /// <summary>
    /// Gets value type.
    /// </summary>
    public new IRType ValueType => base.ValueType;

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
    /// Create TensorConstant from <see cref="Utf8Char"/>.
    /// </summary>
    /// <param name="value">Value.</param>
    public static implicit operator TensorConst(Utf8Char value) => new(Tensor.FromScalar(value));

    /// <summary>
    /// Create TensorConstant from <see cref="string"/>.
    /// </summary>
    /// <param name="value">Value.</param>
    public static implicit operator TensorConst(string value) => new(Tensor.From<char>(value.ToCharArray()));

    public static bool operator ==(TensorConst? left, TensorConst? right) => EqualityComparer<TensorConst>.Default.Equals(left, right);

    public static bool operator !=(TensorConst? left, TensorConst? right) => !(left == right);

    /// <inheritdoc/>
    public override string ToString()
    {
        var type = ValueType switch
        {
            DistributedType dt => dt.TensorType,
            TensorType tt => tt,
            _ => throw new NotSupportedException("Not supported const type: " + ValueType),
        };

        return type switch
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
            _ => $"{type.DType.GetDisplayName()} {type.Shape}",
        };
    }

    /// <inheritdoc/>
    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context)
        => functor.VisitTensorConst(this, context);

    public TensorConst With(Tensor? value = null)
    {
        if (value is null && ValueType is DistributedType dt)
        {
            return new TensorConst(Value, dt.NdSBP, dt.Placement);
        }

        return new TensorConst(value ?? Value);
    }

    /// <inheritdoc/>
    public override bool Equals(object? obj) => Equals(obj as TensorConst);

    /// <inheritdoc/>
    public bool Equals(TensorConst? other) => other is not null && (ReferenceEquals(this, other) || GetHashCode() == other.GetHashCode()) && EqualityComparer<Tensor>.Default.Equals(Value, other.Value);

    /// <inheritdoc/>
    protected override int GetHashCodeCore() => HashCode.Combine(Value);
}
