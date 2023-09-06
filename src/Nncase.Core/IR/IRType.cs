// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.IR;

/// <summary>
/// Expression type.
/// </summary>
public abstract record IRType
{
    /// <summary>
    /// convert the datatype to scalar type.
    /// </summary>
    public static implicit operator IRType(DataType dataType) => TensorType.Scalar(dataType);

    /// <summary>
    /// &lt; 0 − If this is inaccurate than rhs
    /// 0 − If this is the same as rhs
    /// &gt; 0 − If this is more accurate than rhs.
    /// </summary>
    /// <param name="rhs"> ir type. </param>
    /// <returns> int value. </returns>
    public int CompareTo(IRType? rhs) => (this, rhs) switch
    {
        (_, null) => 1,
        (AnyType, var y) => y switch
        {
            AnyType => 0,
            _ => -1,
        },
        (var x, AnyType) => x switch
        {
            AnyType => 0,
            _ => 1,
        },
        (TensorType t1, TensorType t2) => (t1, t2) switch
        {
            var p when p.t1.Shape.Rank != p.t2.Shape.Rank ||
                       p.t1.DType != p.t2.DType => throw new InvalidDataException($"The {this} can't compare with {rhs}"),
            var p => p.t1.Shape.Where(d => d.IsFixed).Count().CompareTo(p.t2.Shape.Where(d => d.IsFixed).Count()),
        },
        (TupleType t1, TupleType t2) => (t1, t2) switch
        {
            var p when p.t1.Count != p.t2.Count => throw new NotSupportedException($"{this} with {rhs}"),
            var p => p.t1.Zip(p.t2).Select(tp => tp.First.CompareTo(tp.Second)).Sum() switch
            {
                0 => 0,
                > 0 => 1,
                < 0 => -1,
            },
        },
        (InvalidType, _) => -1,
        (CallableType t1, CallableType t2) => (t1, t2) switch
        {
            var p when t1.Parameters.Count != t2.Parameters.Count => throw new NotSupportedException($"{this} with {rhs}"),
            var p => p.t1.Parameters.Zip(p.t2.Parameters).Select(tp => tp.First.CompareTo(tp.Second)).Sum() + p.t1.ReturnType.CompareTo(p.t2.ReturnType) switch
            {
                0 => 0,
                > 0 => 1,
                < 0 => -1,
            },
        },
        (var x, var y) => (x, y) switch
        {
            var p when p.x.GetType() == p.y.GetType() => 0,
            _ => throw new NotSupportedException($"{this} with {rhs}"),
        },
    };
}

/// <summary>
/// Any type.
/// </summary>
public sealed record AnyType : IRType
{
    /// <summary>
    /// The default any type instance.
    /// </summary>
    public static readonly AnyType Default = new();

    private AnyType()
    {
    }
}

/// <summary>
/// Invalid type.
/// </summary>
public sealed record InvalidType(string Reason) : IRType;

/// <summary>
/// Tensor type.
/// </summary>
public sealed record TensorType(DataType DType, Shape Shape) : IRType
{
    /// <summary>
    /// Gets a value indicating whether scalar.
    /// </summary>
    public bool IsScalar => Shape.IsScalar;

    /// <summary>
    /// Gets a value indicating whether tensor.
    /// </summary>
    public bool IsTensor => !IsScalar;

    /// <summary>
    /// Initialize a scalar tensor type.
    /// </summary>
    /// <param name="dType">Data type.</param>
    /// <returns>The scalar tensor type.</returns>
    public static TensorType Scalar(DataType dType) => new(dType, Shape.Scalar);

    /// <summary>
    /// Initialize an unranked tensor type.
    /// </summary>
    /// <param name="dType">Data type.</param>
    /// <returns>The unranked tensor type.</returns>
    public static TensorType Unranked(DataType dType) => new(dType, Shape.Unranked);

    /// <summary>
    /// Initialize an invalid tensor type.
    /// </summary>
    /// <param name="dType">Data type.</param>
    /// <returns>The invalid tensor type.</returns>
    public static TensorType Invalid(DataType dType) => new(dType, Shape.Invalid);

    /// <summary>
    /// Initialize an pointer tensor type.
    /// </summary>
    /// <param name="elemType"> the Pointed Element Type.</param>
    /// <returns>the pointer tensor type.</returns>
    public static TensorType Pointer(DataType elemType) => new(new PointerType(elemType, Shape.Scalar), Shape.Scalar);

    /// <inheritdoc/>
    public override string ToString() => DType switch
    {
        PrimType ptype => ptype.GetDisplayName() + (Shape.IsScalar ? string.Empty : Shape.ToString()),
        PointerType { ElemType: PrimType etype } => $"*{etype.GetDisplayName()}",
        ValueType => $"{DType}",
        _ => throw new NotSupportedException(DType.GetType().Name),
    };
}

/// <summary>
/// Tuple type.
/// </summary>
public sealed record TupleType(IRArray<IRType> Fields, bool IsVariadic = false) : IRType, IEnumerable<IRType>, IReadOnlyList<IRType>
{
    /// <summary>
    /// Void type.
    /// </summary>
    public static readonly TupleType Void = new(ImmutableArray<IRType>.Empty);

    /// <summary>
    /// Initializes a new instance of the <see cref="TupleType"/> class.
    /// ctor.
    /// </summary>
    /// <param name="fields">sub fields.</param>
    public TupleType(IEnumerable<IRType> fields)
        : this(fields.ToImmutableArray())
    {
    }

    /// <inheritdoc/>
    public int Count => ((IReadOnlyCollection<IRType>)Fields).Count;

    /// <inheritdoc/>
    public IRType this[int index] => ((IReadOnlyList<IRType>)Fields)[index];

    /// <inheritdoc/>
    public IEnumerator<IRType> GetEnumerator()
    {
        return ((IEnumerable<IRType>)Fields).GetEnumerator();
    }

    /// <inheritdoc/>
    IEnumerator IEnumerable.GetEnumerator()
    {
        return ((IEnumerable)Fields).GetEnumerator();
    }
}

/// <summary>
/// Callable type.
/// </summary>
public sealed record CallableType(IRType ReturnType, IRArray<IRType> Parameters) : IRType;

/// <summary>
/// None type.
/// </summary>
public sealed record NoneType : IRType
{
    /// <summary>
    /// The default any type instance.
    /// </summary>
    public static readonly NoneType Default = new();

    private NoneType()
    {
    }
}
