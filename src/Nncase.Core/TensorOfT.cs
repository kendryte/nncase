// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Buffers;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using CommunityToolkit.HighPerformance.Helpers;
using NetFabric.Hyperlinq;
using Nncase.Buffers;
using Nncase.IR;

namespace Nncase;

/// <summary>
/// Represents a multi-dimensional collection of objects of type T that can be accessed
/// by indices. DenseTensor stores values in a contiguous sequential block of memory
/// where all values are represented.
/// </summary>
/// <typeparam name="T">type contained within the Tensor. Typically a value type such as int, double, float, etc.</typeparam>
public unsafe sealed partial class Tensor<T> : Tensor, IEnumerable<T>, ICollection<T>,
    IReadOnlyCollection<T>, IList<T>, IReadOnlyList<T>, IEquatable<Tensor<T>>
    where T : unmanaged, IEquatable<T>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="Tensor{T}"/> class.
    /// </summary>
    /// <param name="length">Size of the 1-dimensional tensor.</param>
    public Tensor(int length)
        : base(DataType.FromType<T>(), length)
    {
        Buffer = new T[length];
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="Tensor{T}"/> class.
    /// </summary>
    /// <param name="dimensions">An span of integers that represent the size of each dimension of the DenseTensor to create.</param>
    public Tensor(ReadOnlySpan<int> dimensions)
        : base(DataType.FromType<T>(), dimensions)
    {
        Buffer = new T[Length];
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="Tensor{T}"/> class.
    /// </summary>
    /// <param name="buffer">Buffer memory.</param>
    /// <param name="dimensions">An span of integers that represent the size of each dimension of the DenseTensor to create.</param>
    public Tensor(Memory<T> buffer, ReadOnlySpan<int> dimensions)
        : base(DataType.FromType<T>(), dimensions)
    {
        Trace.Assert(Length == buffer.Length);
        Buffer = buffer;
    }

    public static Tensor<T> Empty { get; } = new Tensor<T>(0);

    /// <summary>
    /// Gets memory storing backing values of this tensor.
    /// </summary>
    public Memory<T> Buffer { get; }

    /// <inheritdoc/>
    public override Span<byte> BytesBuffer => MemoryMarshal.AsBytes(Buffer.Span);

    int ICollection<T>.Count => Length;

    bool ICollection<T>.IsReadOnly => false;

    int IReadOnlyCollection<T>.Count => Length;

    T IReadOnlyList<T>.this[int index] => GetValue(index);

    T IList<T>.this[int index]
    {
        get => GetValue(index);
        set => SetValue(index, value);
    }

    /// <summary>
    /// Obtains the value at the specified indices.
    /// </summary>
    /// <param name="indices">A one-dimensional array of integers that represent the indices specifying the
    /// position of the element to get.</param>
    /// <returns>The value at the specified position in this Tensor.</returns>
    public new T this[ReadOnlySpan<int> indices]
    {
        get => GetValue(TensorUtilities.GetIndex(Strides, indices));
        set => SetValue(TensorUtilities.GetIndex(Strides, indices), value);
    }

    /// <summary>
    /// Obtains the value at the specified indices.
    /// </summary>
    /// <param name="indices">A one-dimensional array of integers that represent the indices specifying the
    /// position of the element to get.</param>
    /// <returns>The value at the specified position in this Tensor.</returns>
    public new T this[params int[] indices]
    {
        get => this[indices.AsSpan()];
        set => this[indices.AsSpan()] = value;
    }

    /// <summary>
    /// Create constant from a <see cref="int"/>.
    /// </summary>
    /// <param name="value">Value.</param>
    public static implicit operator Tensor<T>(T value) => FromScalar(value);

    /// <summary>
    /// Creates a shallow copy of this tensor, with new backing storage.
    /// </summary>
    /// <returns>A shallow copy of this tensor.</returns>
    public Tensor<T> Clone()
    {
        var tensor = new Tensor<T>(Dimensions);
        Buffer.CopyTo(tensor.Buffer);
        return tensor;
    }

    /// <summary>
    /// Creates a new Tensor of a different type with the specified dimensions and the
    /// same layout as this tensor with elements initialized to their default value.
    /// </summary>
    /// <typeparam name="TResult">Type contained in the returned Tensor.</typeparam>
    /// <param name="dimensions">An span of integers that represent the size of each dimension of the DenseTensor to create.</param>
    /// <returns>A new tensor with the same layout as this tensor but different type and dimensions.</returns>
    public Tensor<TResult> CloneEmpty<TResult>(ReadOnlySpan<int> dimensions)
        where TResult : unmanaged, IEquatable<TResult>
    {
        return new Tensor<TResult>(dimensions);
    }

    /// <summary>
    /// Gets the value at the specied index, where index is a linearized version of n-dimension
    /// indices using strides.
    /// </summary>
    /// <param name="index">An integer index computed as a dot-product of indices.</param>
    /// <returns>The value at the specified position in this Tensor.</returns>
    public T GetValue(int index)
    {
        return Buffer.Span[index];
    }

    /// <summary>
    /// Reshapes the current tensor to new dimensions, using the same backing storage.
    /// </summary>
    /// <param name="dimensions">An span of integers that represent the size of each dimension of the DenseTensor to create.</param>
    /// <returns>A new tensor that reinterprets backing Buffer of this tensor with different dimensions.</returns>
    public Tensor<T> Reshape(ReadOnlySpan<int> dimensions)
    {
        if (Length != TensorUtilities.GetProduct(dimensions))
        {
            throw new ArgumentException("Length after reshape should remain same.");
        }

        return new Tensor<T>(Buffer, dimensions);
    }

    /// <summary>
    /// Sets the value at the specied index, where index is a linearized version of n-dimension
    /// indices using strides.
    /// </summary>
    /// <param name="index">An integer index computed as a dot-product of indices.</param>
    /// <param name="value">The new value to set at the specified position in this Tensor.</param>
    public void SetValue(int index, T value)
    {
        Buffer.Span[index] = value;
    }

    /// <summary>
    /// Sets all elements in Tensor to value.
    /// </summary>
    /// <param name="value">Value to fill.</param>
    public void Fill(T value)
    {
        Buffer.Span.Fill(value);
    }

    /// <summary>
    /// Determines whether an element is in the <see cref="Tensor{T}"/>.
    /// </summary>
    /// <param name="value">The object to locate in the <see cref="Tensor{T}"/>. The value can be null for reference types.</param>
    /// <returns>true if item is found in the <see cref="Tensor{T}"/>; otherwise, false.</returns>
    public bool Contains(T value)
    {
        return Buffer.Span.Contains(value);
    }

    /// <summary>
    /// Determines the index of a specific item in the <see cref="Tensor{T}"/>.
    /// </summary>
    /// <param name="item">The object to locate in the <see cref="Tensor{T}"/>.</param>
    /// <param name="indices">The index of item if found in the tensor.</param>
    /// <returns>true if item is found in the <see cref="Tensor{T}"/>; otherwise, false.</returns>
    public bool TryGetIndicesOf(T item, Span<int> indices)
    {
        if (indices.Length != Rank)
        {
            throw new ArgumentException(nameof(indices) + " is not sufficient.");
        }

        var index = Buffer.Span.IndexOf(item);
        if (index < 0)
        {
            return false;
        }

        TensorUtilities.GetIndices(Strides, false, index, indices);
        return true;
    }

    /// <inheritdoc/>
    public override string GetArrayString(bool includeWhitespace = true)
    {
        if (Dimensions.IsEmpty)
        {
            return Buffer.Span[0].ToString()!;
        }

        string prefix = TensorOfT.PrefixMap.GetValueOrDefault(typeof(T).TypeHandle, string.Empty);
        string suffix = TensorOfT.SuffixMap.GetValueOrDefault(typeof(T).TypeHandle, string.Empty);

        var builder = new StringBuilder();

        var indices = new int[Rank];
        var innerDimension = Rank - 1;
        var innerLength = Dimensions[innerDimension];

        int indent = 0;
        for (int outerIndex = 0; outerIndex < Length; outerIndex += innerLength)
        {
            TensorUtilities.GetIndices(Strides, false, outerIndex, indices);

            while ((indent < innerDimension) && (indices[indent] == 0))
            {
                // start up
                if (includeWhitespace)
                {
                    Indent(builder, indent, 2);
                }

                indent++;
                builder.Append('{');
                if (includeWhitespace)
                {
                    builder.AppendLine();
                }
            }

            for (int innerIndex = 0; innerIndex < innerLength; innerIndex++)
            {
                indices[innerDimension] = innerIndex;

                if (innerIndex == 0)
                {
                    if (includeWhitespace)
                    {
                        Indent(builder, indent, 2);
                    }

                    if (includeWhitespace)
                    {
                        builder.Append($"[{string.Join(",", indices)}]: {{");
                    }
                    else
                    {
                        builder.Append('{');
                    }
                }
                else
                {
                    builder.Append(',');
                }

                if (!string.IsNullOrEmpty(prefix))
                {
                    builder.Append(prefix);
                }

                builder.Append(this[indices]);
                if (!string.IsNullOrEmpty(suffix))
                {
                    builder.Append(suffix);
                }
            }

            builder.Append('}');

            for (int i = Rank - 2; i >= 0; i--)
            {
                var lastIndex = Dimensions[i] - 1;
                if (indices[i] == lastIndex)
                {
                    // close out
                    --indent;
                    if (includeWhitespace)
                    {
                        builder.AppendLine();
                        Indent(builder, indent, 2);
                    }

                    builder.Append('}');
                }
                else
                {
                    builder.Append(',');
                    if (includeWhitespace)
                    {
                        builder.AppendLine();
                    }

                    break;
                }
            }
        }

        return builder.ToString();
    }

    /// <summary>
    /// Gets an enumerator that enumerates the elements of the <see cref="Tensor{T}"/>.
    /// </summary>
    /// <returns>An enumerator for the current System.Numerics.Tensors.Tensor`1.</returns>
    public new Enumerator GetEnumerator()
    {
        return new Enumerator(this);
    }

    /// <inheritdoc/>
    public override Tensor<TTo> Cast<TTo>(CastMode castMode)
    {
        if (typeof(T) == typeof(TTo))
        {
            return (Tensor<TTo>)(object)this;
        }
        else
        {
            if (castMode == CastMode.Exact)
            {
                throw new InvalidCastException();
            }

            var converter = (ISpanConverter<T, TTo>)CompilerServices.DataTypeService.GetConverter(typeof(T), typeof(TTo));
            var tensor = new Tensor<TTo>(Dimensions);
            converter.ConvertTo(Buffer.Span, tensor.Buffer.Span, castMode);
            return tensor;
        }
    }

    /// <inheritdoc/>
    public override MemoryHandle PinBuffer()
    {
        return Buffer.Pin();
    }

    /// <inheritdoc/>
    public override bool Equals(object? obj)
    {
        if (obj is Tensor<T> other)
        {
            return Equals(other);
        }

        return false;
    }

    /// <inheritdoc/>
    public bool Equals(Tensor<T>? other)
    {
        if (other == null)
        {
            return false;
        }

        if (Rank != other.Rank)
        {
            return false;
        }

        if (!Dimensions.SequenceEqual(other.Dimensions))
        {
            return false;
        }

        return Buffer.Span.SequenceEqual(other.Buffer.Span);
    }

    /// <inheritdoc/>
    public override int GetHashCode()
    {
        return HashCode<T>.Combine(Buffer.Span);
    }

    IEnumerator<T> IEnumerable<T>.GetEnumerator()
    {
        return GetEnumerator();
    }

    void ICollection<T>.Add(T item)
    {
        throw new InvalidOperationException();
    }

    void ICollection<T>.Clear()
    {
        Buffer.Span.Clear();
    }

    bool ICollection<T>.Contains(T item)
    {
        return Contains(item);
    }

    void ICollection<T>.CopyTo(T[] array, int arrayIndex)
    {
        Buffer.Span.CopyTo(array.AsSpan(arrayIndex));
    }

    bool ICollection<T>.Remove(T item)
    {
        throw new InvalidOperationException();
    }

    int IList<T>.IndexOf(T item)
    {
        return Buffer.Span.IndexOf(item);
    }

    void IList<T>.Insert(int index, T item)
    {
        throw new InvalidOperationException();
    }

    void IList<T>.RemoveAt(int index)
    {
        throw new InvalidOperationException();
    }

    /// <inheritdoc/>
    private protected override int CompareTo(object? other, IComparer comparer)
    {
        if (other == null)
        {
            return 1;
        }

        if (other is Tensor<T>)
        {
            return CompareTo((Tensor<T>)other, comparer);
        }

        var array = other as Array;
        if (array != null)
        {
            return CompareTo(array, comparer);
        }

        throw new ArgumentException("Cannot compare.");
    }

    /// <inheritdoc/>
    private protected override bool Equals(object? other, IEqualityComparer comparer)
    {
        if (other == null)
        {
            return false;
        }

        if (other is Tensor<T>)
        {
            return Equals((Tensor<T>)other, comparer);
        }

        var array = other as Array;
        if (array != null)
        {
            return Equals(array, comparer);
        }

        throw new ArgumentException("Cannot compare.");
    }

    /// <inheritdoc/>
    private protected override int GetHashCode(IEqualityComparer comparer)
    {
        HashCode hashCode = default;
        var buffer = Buffer.Span;
        for (int i = 0; i < buffer.Length; i++)
        {
            hashCode.Add(comparer.GetHashCode(buffer[i]));
        }

        return hashCode.ToHashCode();
    }

    private protected override IEnumerator GetEnumeratorCore()
    {
        return GetEnumerator();
    }

    private protected override void CopyToCore(Array array, int index)
    {
        if (array is T[] arr)
        {
            Buffer.Span.CopyTo(arr.AsSpan(index));
        }
        else
        {
            if (array == null)
            {
                throw new ArgumentNullException(nameof(array));
            }
            else
            {
                if (array.Length < index + Length)
                {
                    throw new ArgumentException("Array is not sufficient.");
                }

                var buffer = Buffer;
                for (int i = 0; i < buffer.Length; i++)
                {
                    array.SetValue(buffer.Span[i], index + i);
                }
            }
        }
    }

    private protected override object GetValueCore(int index)
    {
        return GetValue(index);
    }

    private protected override void SetValueCore(int index, object? value)
    {
        SetValue(index, (T)Convert.ChangeType(value, typeof(T))!);
    }

    private static void Indent(StringBuilder builder, int tabs, int spacesPerTab = 4)
    {
        for (int tab = 0; tab < tabs; tab++)
        {
            for (int space = 0; space < spacesPerTab; space++)
            {
                builder.Append(' ');
            }
        }
    }

    private int CompareTo(Tensor<T> other, IComparer comparer)
    {
        if (Rank != other.Rank)
        {
            throw new ArgumentException("Different ranks.");
        }

        if (Dimensions != other.Dimensions)
        {
            throw new ArgumentException("Different dimensions.");
        }

        var bufferA = Buffer;
        var bufferB = other.Buffer;
        int result = 0;

        for (int i = 0; i < bufferA.Length; i++)
        {
            result = comparer.Compare(bufferA.Span[i], bufferB.Span[i]);
            if (result != 0)
            {
                break;
            }
        }

        return result;
    }

    private int CompareTo(Array other, IComparer comparer)
    {
        if (Rank != other.Rank)
        {
            throw new ArgumentException("Different ranks.");
        }

        var dimensions = Dimensions;
        for (int i = 0; i < dimensions.Length; i++)
        {
            if (dimensions[i] != other.GetLength(i))
            {
                throw new ArgumentException("Different dimensions.");
            }
        }

        var bufferA = Buffer;
        var indices = new int[Rank];
        int result = 0;

        for (int i = 0; i < bufferA.Length; i++)
        {
            TensorUtilities.GetIndices(Strides, false, i, indices);
            result = comparer.Compare(bufferA.Span[i], other.GetValue(indices));
            if (result != 0)
            {
                break;
            }
        }

        return result;
    }

    private bool Equals(Tensor<T> other, IEqualityComparer comparer)
    {
        if (Rank != other.Rank)
        {
            throw new ArgumentException("Different ranks.");
        }

        if (Dimensions != other.Dimensions)
        {
            throw new ArgumentException("Different dimensions.");
        }

        var bufferA = Buffer;
        var bufferB = other.Buffer;

        for (int i = 0; i < bufferA.Length; i++)
        {
            if (!comparer.Equals(bufferA.Span[i], bufferB.Span[i]))
            {
                return false;
            }
        }

        return true;
    }

    private bool Equals(Array other, IEqualityComparer comparer)
    {
        if (Rank != other.Rank)
        {
            throw new ArgumentException("Different ranks.");
        }

        var dimensions = Dimensions;
        for (int i = 0; i < dimensions.Length; i++)
        {
            if (dimensions[i] != other.GetLength(i))
            {
                throw new ArgumentException("Different dimensions.");
            }
        }

        var bufferA = Buffer;
        var indices = new int[Rank];

        for (int i = 0; i < bufferA.Length; i++)
        {
            TensorUtilities.GetIndices(Strides, false, i, indices);
            if (!comparer.Equals(bufferA.Span[i], other.GetValue(indices)))
            {
                return false;
            }
        }

        return true;
    }

    /// <summary>
    /// The type that implements enumerators for System.Numerics.Tensors.Tensor`1 instances.
    /// </summary>
    public struct Enumerator : IEnumerator<T>, IEnumerator, IDisposable
    {
        private readonly Tensor<T> _tensor;

        private int _index;

        internal Enumerator(Tensor<T> tensor)
        {
            _tensor = tensor;
            _index = 0;
            Current = default;
        }

        /// <inheritdoc/>
        public T Current { get; private set; }

        object? IEnumerator.Current => Current;

        /// <inheritdoc/>
        public bool MoveNext()
        {
            if (_index < _tensor.Length)
            {
                Current = _tensor.GetValue(_index);
                _index++;
                return true;
            }

            Current = default;
            return false;
        }

        /// <inheritdoc/>
        public void Reset()
        {
            _index = 0;
            Current = default;
        }

        /// <inheritdoc/>
        public void Dispose()
        {
        }
    }
}

internal sealed class TensorOfT
{
    /// <summary>
    /// The array to string prefix map.
    /// </summary>
    public static readonly Dictionary<System.RuntimeTypeHandle, string> PrefixMap = new()
    {
        { typeof(Half).TypeHandle, "(Half)" },
        { typeof(BFloat16).TypeHandle, "(BFloat16)" },
    };

    /// <summary>
    /// The array to string suffix map.
    /// </summary>
    public static readonly Dictionary<System.RuntimeTypeHandle, string> SuffixMap = new()
    {
        { typeof(float).TypeHandle, "f" },
        { typeof(double).TypeHandle, "d" },
        { typeof(long).TypeHandle, "L" },
        { typeof(uint).TypeHandle, "U" },
        { typeof(ulong).TypeHandle, "UL" },
    };
}
