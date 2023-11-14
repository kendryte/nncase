// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Buffers;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Toolkit.HighPerformance;
using Nncase.Buffers;
using Nncase.IR;

namespace Nncase;

/// <summary>
/// Cast mode.
/// </summary>
public enum CastMode
{
    /// <summary>
    /// Cast as possible.
    /// <remarks>
    /// rename to kdefault avoid cpp runtime compile error
    /// </remarks>
    /// </summary>
    KDefault,

    /// <summary>
    /// Cast exactly.
    /// </summary>
    Exact,

    /// <summary>
    /// Check overflow.
    /// </summary>
    CheckOverflow,

    /// <summary>
    /// Reinterpret cast.
    /// </summary>
    Reinterpret,
}

/// <summary>
/// Tensor.
/// </summary>
[DebuggerDisplay("{GetArrayString(false)}")]
public abstract partial class Tensor : IStructuralComparable, IStructuralEquatable, IEnumerable, ICollection, IList
{
    private static readonly MethodInfo _tensorCreateFromBytesFunc =
        typeof(Tensor).GetMethod(nameof(CreateTensorFromBytesImpl), BindingFlags.Static | BindingFlags.NonPublic)!;

    private static readonly MethodInfo _tensorCreateFromArrayFunc =
        typeof(Tensor).GetMethod(nameof(CreateTensorFromArrayImpl), BindingFlags.Static | BindingFlags.NonPublic)!;

    private static readonly MethodInfo _tensorCastFunc =
        typeof(Tensor).GetMethod(nameof(Cast))!;

    private readonly int[] _dimensions;
    private readonly int[] _strides;

    /// <summary>
    /// Initializes a new instance of the <see cref="Tensor"/> class.
    /// </summary>
    /// <param name="elementType">Element type.</param>
    /// <param name="length">Size of the 1-dimensional tensor.</param>
    internal Tensor(DataType elementType, int length)
    {
        ElementType = elementType;
        Shape = new Shape(length);
        Length = length;
        _dimensions = new[] { length };
        _strides = TensorUtilities.GetStrides(_dimensions);
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="Tensor"/> class.
    /// </summary>
    /// <param name="elementType">Element type.</param>
    /// <param name="dimensions">An span of integers that represent the size of each dimension of the DenseTensor to create.</param>
    internal Tensor(DataType elementType, ReadOnlySpan<int> dimensions)
    {
        ElementType = elementType;
        _dimensions = dimensions.ToArray();
        Shape = dimensions.IsEmpty ? Shape.Scalar : new Shape(_dimensions);
        Length = (int)TensorUtilities.GetProduct(dimensions);
        _strides = TensorUtilities.GetStrides(dimensions);
    }

    /// <summary>
    /// Gets element type.
    /// </summary>
    public DataType ElementType { get; }

    /// <summary>
    /// Gets dimensions.
    /// </summary>
    public ReadOnlySpan<int> Dimensions => _dimensions;

    /// <summary>
    /// Gets strides.
    /// </summary>
    public ReadOnlySpan<int> Strides => _strides;

    /// <summary>
    /// Gets shape.
    /// </summary>
    public Shape Shape { get; }

    /// <summary>
    /// Gets rank of the tensor: number of dimensions.
    /// </summary>
    public int Rank => Dimensions.Length;

    /// <summary>
    /// Gets total length.
    /// </summary>
    public int Length { get; }

    /// <summary>
    /// Gets bytes buffer.
    /// </summary>
    public abstract Span<byte> BytesBuffer { get; }

    int ICollection.Count => Length;

    bool ICollection.IsSynchronized => false;

    object ICollection.SyncRoot => this;

    bool IList.IsFixedSize => true;

    bool IList.IsReadOnly => false;

    object? IList.this[int index]
    {
        get => GetValueCore(index);
        set => SetValueCore(index, value);
    }

    /// <summary>
    /// Obtains the value at the specified indices.
    /// </summary>
    /// <param name="indices">A one-dimensional array of integers that represent the indices specifying the
    /// position of the element to get.</param>
    /// <returns>The value at the specified position in this Tensor.</returns>
    public object this[ReadOnlySpan<int> indices]
    {
        get => GetValueCore(TensorUtilities.GetIndex(Strides, indices));
        set => SetValueCore(TensorUtilities.GetIndex(Strides, indices), value);
    }

    /// <summary>
    /// Obtains the value at the specified indices.
    /// </summary>
    /// <param name="indices">A one-dimensional array of integers that represent the indices specifying the
    /// position of the element to get.</param>
    /// <returns>The value at the specified position in this Tensor.</returns>
    public object this[params int[] indices]
    {
        get => this[indices.AsSpan()];
        set => this[indices.AsSpan()] = value;
    }

    /// <summary>
    /// Create a scalar tensor from a scalar.
    /// </summary>
    /// <typeparam name="T">CLR type.</typeparam>
    /// <param name="value">Value.</param>
    /// <returns>Created tensor.</returns>
    public static Tensor<T> FromScalar<T>(T value)
        where T : unmanaged, IEquatable<T>
    {
        var tensor = new Tensor<T>(ReadOnlySpan<int>.Empty);
        tensor[0] = value;
        return tensor;
    }

    /// <summary>
    /// Create a 1-D tensor from a scalar.
    /// </summary>
    /// <typeparam name="T">CLR type.</typeparam>
    /// <param name="value">Value.</param>
    /// <param name="length">Fill length.</param>
    /// <returns>Created tensor.</returns>
    public static Tensor<T> FromScalar<T>(T value, int length)
        where T : unmanaged, IEquatable<T>
    {
        var tensor = new Tensor<T>(MemoryMarshal.CreateReadOnlySpan(ref length, 1));
        tensor.Fill(value);
        return tensor;
    }

    /// <summary>
    /// Create a N-D tensor from a scalar.
    /// </summary>
    /// <typeparam name="T">CLR type.</typeparam>
    /// <param name="value">Value.</param>
    /// <param name="dimensions">Fill dimensions.</param>
    /// <returns>Created tensor.</returns>
    public static Tensor<T> FromScalar<T>(T value, ReadOnlySpan<int> dimensions)
        where T : unmanaged, IEquatable<T>
    {
        var tensor = new Tensor<T>(dimensions);
        tensor.Fill(value);
        return tensor;
    }

    /// <summary>
    /// Create tensor from a range.
    /// </summary>
    /// <param name="start">Start value.</param>
    /// <param name="count">Count.</param>
    /// <returns>Created tensor.</returns>
    public static Tensor<int> FromRange(int start, int count)
    {
        var tensor = new Tensor<int>(MemoryMarshal.CreateReadOnlySpan(ref count, 1));
        var buffer = tensor.Buffer.Span;
        for (int i = 0; i < count; i++)
        {
            buffer[i] = start + i;
        }

        return tensor;
    }

    /// <summary>
    /// Create tensor from a memory, Set the shape as [n].
    /// </summary>
    /// <typeparam name="T">CLR type.</typeparam>
    /// <param name="memory">Memory.</param>
    /// <returns>Created tensor.</returns>
    public static Tensor<T> From<T>(Memory<T> memory)
        where T : unmanaged, IEquatable<T>
    {
        var dim = memory.Length;
        return new Tensor<T>(memory, MemoryMarshal.CreateReadOnlySpan(ref dim, 1));
    }

    /// <summary>
    /// Create tensor from a memory, Set the shape as provided.
    /// </summary>
    /// <typeparam name="T">CLR type.</typeparam>
    /// <param name="memory">Memory.</param>
    /// <param name="dimensions">Dimensions.</param>
    /// <returns>Created tensor.</returns>
    public static Tensor<T> From<T>(Memory<T> memory, ReadOnlySpan<int> dimensions)
        where T : unmanaged, IEquatable<T>
    {
        return new Tensor<T>(memory, dimensions);
    }

    /// <summary>
    /// Create tensor from an array, Set the shape as [n].
    /// </summary>
    /// <typeparam name="T">CLR type.</typeparam>
    /// <param name="array">Array.</param>
    /// <returns>Created tensor.</returns>
    public static Tensor<T> From<T>(T[] array)
        where T : unmanaged, IEquatable<T>
    {
        return From(array.AsMemory());
    }

    /// <summary>
    /// Create tensor from an array, Set the shape as provided.
    /// </summary>
    /// <typeparam name="T">CLR type.</typeparam>
    /// <param name="array">Array.</param>
    /// <param name="dimensions">Dimensions.</param>
    /// <returns>Created tensor.</returns>
    public static Tensor<T> From<T>(T[] array, ReadOnlySpan<int> dimensions)
        where T : unmanaged, IEquatable<T>
    {
        return From(array.AsMemory(), dimensions);
    }

    /// <summary>
    /// Create tensor from a bytes span, Set the shape as provided.
    /// </summary>
    /// <typeparam name="T">CLR type.</typeparam>
    /// <param name="memory">Bytes memory.</param>
    /// <param name="dimensions">Dimensions.</param>
    /// <returns>Created tensor.</returns>
    public static Tensor<T> FromBytes<T>(Memory<byte> memory, ReadOnlySpan<int> dimensions)
        where T : unmanaged, IEquatable<T>
    {
        return new Tensor<T>(memory.Cast<byte, T>(), dimensions);
    }

    /// <summary>
    /// Create tensor from a bytes memory, Set the shape as provided.
    /// </summary>
    /// <param name="type">Data type.</param>
    /// <param name="memory">Bytes memory.</param>
    /// <param name="dimensions">Dimensions.</param>
    /// <returns>Created tensor.</returns>
    public static Tensor FromBytes(DataType type, Memory<byte> memory, ReadOnlySpan<int> dimensions)
    {
        return (Tensor)_tensorCreateFromBytesFunc.MakeGenericMethod(type.CLRType).Invoke(null, new object[] { memory, dimensions.ToArray() })!;
    }

    /// <summary>
    /// Create tensor from a bytes memory, Set the shape as provided.
    /// </summary>
    /// <param name="type">Tensor type.</param>
    /// <param name="buffer">Bytes memory.</param>
    /// <returns>Created tensor.</returns>
    public static Tensor FromBytes(TensorType type, Memory<byte> buffer)
    {
        return FromBytes(type.DType, buffer, type.Shape.ToValueArray());
    }

    /// <summary>
    /// Create tensor from an array.
    /// </summary>
    /// <param name="array">Array.</param>
    /// <returns>Created tensor.</returns>
    public static unsafe Tensor FromArray(Array array)
    {
        var elemType = array.GetType().GetElementType()!;
        var dims = new int[array.Rank];
        for (int i = 0; i < array.Rank; i++)
        {
            dims[i] = array.GetLength(i);
        }

        return (Tensor)_tensorCreateFromArrayFunc.MakeGenericMethod(elemType).Invoke(null, new object[] { array, dims })!;
    }

    /// <summary>
    /// Create tensor from a ulong address.
    /// </summary>
    /// <typeparam name="T">CLR type.</typeparam>
    /// <param name="value">addr value.</param>
    /// <returns>Created tensor.</returns>
    public static Tensor<Pointer<T>> FromPointer<T>(ulong value)
      where T : unmanaged, IEquatable<T>
    {
        return FromScalar<Pointer<T>>(new Pointer<T>(value));
    }

    /// <summary>
    /// Create tensor from a ulong address.
    /// </summary>
    /// <param name="value">addr value.</param>
    /// <param name="elemType">pointed type.</param>
    /// <returns>Created tensor.</returns>
    public static Tensor FromPointer(ulong value, DataType elemType)
    {
        return FromBytes(TensorType.Scalar(new PointerType(elemType)), BitConverter.GetBytes(value));
    }

    /// <summary>
    /// convert Const To Tensor.
    /// </summary>
    /// <param name="const"> const.</param>
    /// <returns> Tensor. </returns>
    public static Tensor FromConst(Const @const) => @const switch
    {
        TensorConst tc => tc.Value,
        TupleConst => throw new InvalidOperationException("Can't Convert TupleConst To Tensor!"),
        _ => throw new NotSupportedException(@const.GetType().Name),
    };

    /// <summary>
    /// convert Const To Tensor T.
    /// </summary>
    /// <typeparam name="T">unmanaged type.</typeparam>
    /// <param name="const">const.</param>
    /// <param name="castMode">castmode.</param>
    /// <returns>Tensor{T}.</returns>
    public static Tensor<T> FromConst<T>(Const @const, CastMode castMode = CastMode.KDefault)
      where T : unmanaged, IEquatable<T>
      => FromConst(@const).Cast<T>(castMode);

    /// <summary>
    /// Return a tensor of given shape and type, filled with zeros.
    /// </summary>
    /// <typeparam name="T">unmanaged type.</typeparam>
    /// <param name="dimensions">dimensions.</param>
    /// <returns>Tensor{T}.</returns>
    public static Tensor Zeros<T>(ReadOnlySpan<int> dimensions)
        where T : unmanaged, IEquatable<T>
    {
        var value = (T)Convert.ChangeType(0, typeof(T));
        return Tensor.FromScalar<T>(value, dimensions);
    }

    /// <summary>
    /// Return a tensor of given shape and type, filled with ones.
    /// </summary>
    /// <typeparam name="T">unmanaged type.</typeparam>
    /// <param name="dimensions">dimensions.</param>
    /// <returns>Tensor{T}.</returns>
    public static Tensor Ones<T>(ReadOnlySpan<int> dimensions)
        where T : unmanaged, IEquatable<T>
    {
        var value = (T)Convert.ChangeType(1, typeof(T));
        return Tensor.FromScalar<T>(value, dimensions);
    }

    /// <summary>
    /// Cast to typed tensor.
    /// </summary>
    /// <typeparam name="T">Element type.</typeparam>
    /// <param name="castMode">Cast mode.</param>
    /// <returns>Typed tensor.</returns>
    public abstract Tensor<T> Cast<T>(CastMode castMode = CastMode.KDefault)
        where T : unmanaged, IEquatable<T>;

    /// <summary>
    /// <see cref="Cast{T}(CastMode)"/>.
    /// </summary>
    public Tensor CastTo(DataType type, CastMode castMode = CastMode.KDefault)
    {
        var tensor = (Tensor)_tensorCastFunc.MakeGenericMethod(type.CLRType).Invoke(this, new object[] { castMode })!;
        return tensor;
    }

    /// <summary>
    /// Pin buffer.
    /// </summary>
    /// <returns>Memory handle.</returns>
    public abstract MemoryHandle PinBuffer();

    /// <inheritdoc/>
    public IEnumerator GetEnumerator()
    {
        return GetEnumeratorCore();
    }

    /// <summary>
    /// Get array string.
    /// </summary>
    /// <param name="includeWhitespace">Include whitespace.</param>
    /// <returns>String of this tensor.</returns>
    public abstract string GetArrayString(bool includeWhitespace = true);

    int IStructuralComparable.CompareTo(object? other, IComparer comparer)
    {
        return CompareTo(other, comparer);
    }

    bool IStructuralEquatable.Equals(object? other, IEqualityComparer comparer)
    {
        return Equals(other, comparer);
    }

    int IStructuralEquatable.GetHashCode(IEqualityComparer comparer)
    {
        return GetHashCode(comparer);
    }

    void ICollection.CopyTo(Array array, int index)
    {
        CopyToCore(array, index);
    }

    int IList.Add(object? value)
    {
        throw new InvalidOperationException();
    }

    void IList.Clear()
    {
        BytesBuffer.Clear();
    }

    bool IList.Contains(object? value)
    {
        throw new NotImplementedException();
    }

    int IList.IndexOf(object? value)
    {
        throw new NotImplementedException();
    }

    void IList.Insert(int index, object? value)
    {
        throw new InvalidOperationException();
    }

    void IList.Remove(object? value)
    {
        throw new InvalidOperationException();
    }

    void IList.RemoveAt(int index)
    {
        throw new InvalidOperationException();
    }

    private protected abstract int CompareTo(object? other, IComparer comparer);

    private protected abstract bool Equals(object? other, IEqualityComparer comparer);

    private protected abstract int GetHashCode(IEqualityComparer comparer);

    private protected abstract IEnumerator GetEnumeratorCore();

    private protected abstract void CopyToCore(Array array, int index);

    private protected abstract object GetValueCore(int index);

    private protected abstract void SetValueCore(int index, object? value);

    private static Tensor CreateTensorFromBytesImpl<T>(Memory<byte> buffer, int[] dimensions)
        where T : unmanaged, IEquatable<T>
    {
        return new Tensor<T>(buffer.Cast<byte, T>(), dimensions);
    }

    private static Tensor CreateTensorFromArrayImpl<T>(Array array, int[] dimensions)
        where T : unmanaged, IEquatable<T>
    {
        var mmgr = new ArrayMemoryManager<T>(array);
        return new Tensor<T>(mmgr.Memory, dimensions);
    }
}
