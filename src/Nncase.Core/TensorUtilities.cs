﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.Utilities;

namespace Nncase;

/// <summary>
/// Array helper.
/// </summary>
public static class TensorUtilities
{
    private const int StackallocMax = 16;

    private enum SliceStatus : uint
    {
        IsFull,
        IsSlice,
        IsSliceFull, // shape [10,10] like [[0,1), [0,10)]
        IsInvalid,
    }

    /// <summary>
    /// get the product from the start index on the dimensions.
    /// </summary>
    public static T GetProductGeneric<T>(ReadOnlySpan<T> dimensions, int startIndex = 0)
        where T : struct, ISignedNumber<T>, IComparisonOperators<T, T, bool>
    {
        if (dimensions.Length == 0)
        {
            return T.One;
        }

        T product = T.One;
        for (int i = startIndex; i < dimensions.Length; i++)
        {
            // we use a long which should be much larger than is ever used here,
            // but still force checked
            checked
            {
                product *= dimensions[i];
            }
        }

        return product;
    }

    /// <summary>
    /// get the product from the start index on the dimensions.
    /// </summary>
    public static int GetProduct(ReadOnlySpan<int> dimensions, int startIndex = 0) => GetProductGeneric(dimensions, startIndex);

    /// <summary>
    /// get the product from the start index on the dimensions.
    /// </summary>
    public static long GetProduct(ReadOnlySpan<long> dimensions, int startIndex = 0) => GetProductGeneric(dimensions, startIndex);

    /// <summary>
    /// Get the Expr Product.
    /// </summary>
    public static Dimension GetProduct(ReadOnlySpan<Dimension> dimensions, int startIndex = 0)
    {
        if (dimensions.Length == 0)
        {
            return 1L;
        }

        Dimension product = 1L;
        for (int i = startIndex; i < dimensions.Length; i++)
        {
            var dimension = dimensions[i];
            product *= dimension;
        }

        return product;
    }

    public static bool IsAscending(ReadOnlySpan<long> values)
    {
        for (int i = 1; i < values.Length; i++)
        {
            if (values[i] < values[i - 1])
            {
                return false;
            }
        }

        return true;
    }

    public static bool IsDescending(ReadOnlySpan<long> values)
    {
        for (int i = 1; i < values.Length; i++)
        {
            if (values[i] > values[i - 1])
            {
                return false;
            }
        }

        return true;
    }

    /// <summary>
    /// Gets the set of strides that can be used to calculate the offset of n-dimensions in a 1-dimensional layout.
    /// </summary>
    public static T[] GetStridesGeneric<T>(ReadOnlySpan<T> dimensions, bool reverseStride = false)
        where T : struct, ISignedNumber<T>
    {
        if (dimensions.IsEmpty)
        {
            return Array.Empty<T>();
        }

        var strides = new T[dimensions.Length];

        T stride = T.One;
        if (reverseStride)
        {
            for (int i = 0; i < strides.Length; i++)
            {
                strides[i] = stride;
                stride *= dimensions[i];
            }
        }
        else
        {
            for (int i = strides.Length - 1; i >= 0; i--)
            {
                strides[i] = stride;
                stride *= dimensions[i];
            }
        }

        return strides;
    }

    /// <summary>
    /// Gets the set of strides that can be used to calculate the offset of n-dimensions in a 1-dimensional layout.
    /// </summary>
    public static int[] GetStrides(ReadOnlySpan<int> dimensions, bool reverseStride = false) => GetStridesGeneric(dimensions, reverseStride);

    /// <summary>
    /// Gets the set of strides that can be used to calculate the offset of n-dimensions in a 1-dimensional layout.
    /// </summary>
    public static long[] GetStrides(ReadOnlySpan<long> dimensions, bool reverseStride = false) => GetStridesGeneric(dimensions, reverseStride);

    /// <summary>
    /// get strides.
    /// </summary>
    public static Dimension[] GetStrides(ReadOnlySpan<Dimension> dimensions, bool reverseStride = false)
    {
        if (dimensions.IsEmpty)
        {
            return Array.Empty<Dimension>();
        }

        var strides = new Dimension[dimensions.Length];

        Dimension stride = 1;
        if (reverseStride)
        {
            for (int i = 0; i < strides.Length; i++)
            {
                strides[i] = stride;
                stride *= dimensions[i];
            }
        }
        else
        {
            for (int i = strides.Length - 1; i >= 0; i--)
            {
                strides[i] = stride;
                stride *= dimensions[i];
            }
        }

        return strides;
    }

    public static void SplitStrides(long[] strides, int[] splitAxes, long[] newStrides, long stridesOffset, long[] splitStrides, long splitStridesOffset)
    {
        int newStrideIndex = 0;
        for (int i = 0; i < strides.Length; i++)
        {
            long stride = strides[i];
            bool isSplit = false;
            for (int j = 0; j < splitAxes.Length; j++)
            {
                if (splitAxes[j] == i)
                {
                    splitStrides[splitStridesOffset + j] = stride;
                    isSplit = true;
                    break;
                }
            }

            if (!isSplit)
            {
                newStrides[stridesOffset + newStrideIndex++] = stride;
            }
        }
    }

    /// <summary>
    /// Calculates the 1-d index for n-d indices in layout specified by strides.
    /// </summary>
    public static T GetIndexGeneric<T>(ReadOnlySpan<T> strides, ReadOnlySpan<T> indices, int startFromDimension = 0)
        where T : struct, IBinaryNumber<T>, IComparisonOperators<T, T, bool>
    {
        // Scalar
        if (strides.Length == 0)
        {
            if (indices.Length != 0)
            {
                throw new ArgumentOutOfRangeException(nameof(indices));
            }

            return T.Zero;
        }

        if (strides.Length != indices.Length)
        {
            throw new ArgumentOutOfRangeException(nameof(indices));
        }

        T index = T.Zero;
        for (int i = startFromDimension; i < indices.Length; i++)
        {
            index += strides[i] * indices[i];
        }

        return index;
    }

    /// <summary>
    /// Calculates the 1-d index for n-d indices in layout specified by strides.
    /// </summary>
    public static int GetIndex(ReadOnlySpan<int> strides, ReadOnlySpan<int> indices, int startFromDimension = 0) => GetIndexGeneric(strides, indices, startFromDimension);

    /// <summary>
    /// Calculates the 1-d index for n-d indices in layout specified by strides.
    /// </summary>
    public static long GetIndex(ReadOnlySpan<long> strides, ReadOnlySpan<long> indices, int startFromDimension = 0) => GetIndexGeneric(strides, indices, startFromDimension);

    /// <summary>
    /// get index.
    /// </summary>
    public static IR.Expr GetIndex(ReadOnlySpan<IR.Expr> strides, ReadOnlySpan<IR.Expr> indices, int startFromDimension = 0)
    {
        // Scalar
        if (strides.Length == 0)
        {
            if (indices.Length != 0)
            {
                throw new ArgumentOutOfRangeException(nameof(indices));
            }

            return 0L;
        }

        if (strides.Length != indices.Length)
        {
            throw new ArgumentOutOfRangeException(nameof(indices));
        }

        IR.Expr index = 0L;
        for (int i = startFromDimension; i < indices.Length; i++)
        {
            index += strides[i] * indices[i];
        }

        return index;
    }

    /// <summary>
    /// Calculates the n-d indices from the 1-d index in a layout specificed by strides.
    /// </summary>
    public static void GetIndices(ReadOnlySpan<long> strides, bool reverseStride, long index, long[] indices, int startFromDimension = 0)
    {
        Trace.Assert(reverseStride ? IsAscending(strides) : IsDescending(strides), "Index decomposition requires ordered strides");
        if (strides.Length != indices.Length)
        {
            throw new ArgumentOutOfRangeException(nameof(indices));
        }

        long remainder = index;
        for (int i = startFromDimension; i < strides.Length; i++)
        {
            // reverse the index for reverseStride so that we divide by largest stride first
            var nIndex = reverseStride ? strides.Length - 1 - i : i;

            var stride = strides[nIndex];
            indices[nIndex] = remainder / stride;
            remainder %= stride;
        }
    }

    /// <summary>
    /// Calculates the n-d indices from the 1-d index in a layout specificed by strides.
    /// </summary>
    public static void GetIndices(ReadOnlySpan<long> strides, bool reverseStride, long index, Span<long> indices, int startFromDimension = 0)
    {
        Trace.Assert(reverseStride ? IsAscending(strides) : IsDescending(strides), "Index decomposition requires ordered strides");
        if (strides.Length != indices.Length)
        {
            throw new ArgumentOutOfRangeException(nameof(indices));
        }

        long remainder = index;
        for (int i = startFromDimension; i < strides.Length; i++)
        {
            // reverse the index for reverseStride so that we divide by largest stride first
            var nIndex = reverseStride ? strides.Length - 1 - i : i;

            var stride = strides[nIndex];
            indices[nIndex] = remainder / stride;
            remainder %= stride;
        }
    }

    /// <summary>
    /// Takes an 1-d index over n-d sourceStrides and recalculates it assuming same n-d coordinates over a different n-d strides.
    /// </summary>
    public static long TransformIndexByStrides(long index, long[] sourceStrides, bool sourceReverseStride, long[] transformStrides)
    {
        Trace.Assert(index >= 0);
        Trace.Assert(sourceReverseStride ? IsAscending(sourceStrides) : IsDescending(sourceStrides), "Index decomposition requires ordered strides");
        if (sourceStrides.Length != transformStrides.Length)
        {
            throw new ArgumentOutOfRangeException(nameof(transformStrides));
        }

        long transformIndex = 0;
        long remainder = index;

        for (int i = 0; i < sourceStrides.Length; i++)
        {
            // reverse the index for reverseStride so that we divide by largest stride first
            var nIndex = sourceReverseStride ? sourceStrides.Length - 1 - i : i;

            var sourceStride = sourceStrides[nIndex];
            var transformStride = transformStrides[nIndex];

            transformIndex += transformStride * (remainder / sourceStride);
            remainder %= sourceStride;
        }

        return transformIndex;
    }

    /// <summary>
    /// check this dimension and strides is contiguous.
    /// </summary>
    public static bool IsContiguous(ReadOnlySpan<long> dimensions, ReadOnlySpan<long> strides)
    {
        return System.Collections.StructuralComparisons.StructuralEqualityComparer.Equals(GetStrides(dimensions), strides.ToArray());
    }

    /// <summary>
    /// from end to front calculate the number of contiguous dimensions.
    /// </summary>
    public static int GetContiguousDims(ReadOnlySpan<long> dimensions, ReadOnlySpan<long> strides)
    {
        var def_strides = GetStrides(dimensions);
        for (int i = strides.Length - 1; i >= 0; --i)
        {
            if (strides[i] != def_strides[i])
            {
                return dimensions.Length - i - 1;
            }
        }

        return dimensions.Length;
    }

    /// <summary>
    /// check the dimensions selected range is contiguous.
    /// </summary>
    public static bool IsContiguousSlice(ReadOnlySpan<long> dimensions, ReadOnlySpan<System.Range> slices, out int contiguousStart)
    {
        if (dimensions.Length != slices.Length)
        {
            contiguousStart = slices.Length - 1;
            return false;
        }

        SliceStatus status = SliceStatus.IsFull;
        for (int i = dimensions.Length - 1; i >= 0; i--)
        {
            var start = slices[i].Start.IsFromEnd ? dimensions[i] - slices[i].Start.Value : slices[i].Start.Value;
            var end = slices[i].End.IsFromEnd ? dimensions[i] - slices[i].End.Value : slices[i].End.Value;

            status = (end - start) switch
            {
                // is full
                long x when x == dimensions[i] => status switch
                {
                    SliceStatus.IsSlice => x == 1 ?
                                                    SliceStatus.IsSlice :
                                                    SliceStatus.IsInvalid,
                    SliceStatus.IsSliceFull => x == 1 ?
                                                    SliceStatus.IsSliceFull :
                                                    SliceStatus.IsInvalid,
                    _ => SliceStatus.IsFull,
                },

                // when has
                long x when x > 0 && x < dimensions[i] => status switch
                {
                    SliceStatus.IsSlice => x == 1 ?
                                                SliceStatus.IsSlice :
                                                SliceStatus.IsInvalid,
                    SliceStatus.IsSliceFull => x == 1 ?
                                                SliceStatus.IsSliceFull :
                                                SliceStatus.IsInvalid,
                    SliceStatus.IsFull => SliceStatus.IsSliceFull,
                    _ => SliceStatus.IsSlice,
                },
                _ => throw new NotSupportedException(),
            };
            if (status == SliceStatus.IsInvalid)
            {
                contiguousStart = i + 1;
                return false;
            }
        }

        contiguousStart = 0;
        return true;
    }

    public static bool IsContiguousSlice(ReadOnlySpan<long> dimensions, ReadOnlySpan<System.Range> slices) => IsContiguousSlice(dimensions, slices, out _);

    public static long[] ToLongs(this ReadOnlySpan<int> ints)
    {
        var longs = new long[ints.Length];
        for (int i = 0; i < longs.Length; i++)
        {
            longs[i] = ints[i];
        }

        return longs;
    }

    public static long[] ToLongs(this int[] ints) => ToLongs((ReadOnlySpan<int>)ints);

    public static int[] ToInts(this ReadOnlySpan<long> longs)
    {
        var ints = new int[longs.Length];
        for (int i = 0; i < ints.Length; i++)
        {
            ints[i] = checked((int)longs[i]);
        }

        return ints;
    }

    public static int[] ToInts(this long[] longs) => ToInts((ReadOnlySpan<long>)longs);

    public static long GetSize(ReadOnlySpan<long> shapes, ReadOnlySpan<long> strides, int elementSize)
    {
        long max_stride = 1, max_shape = 1;
        for (int i = 0; i < shapes.Length; i++)
        {
            if ((shapes[i] == 1 ? 0 : strides[i]) >= max_stride)
            {
                max_stride = strides[i];
                max_shape = shapes[i];
            }
        }

        long size = max_stride * max_shape;
        return size * elementSize;
    }

    public static (long MaxSize, long[] Strides) GetTensorMaxSizeAndStrides(TensorType tensorType, DistributedType? distributedType)
    {
        long[] dims;
        long[] strides;
        if (distributedType is null)
        {
            dims = CompilerServices.GetMaxShape(tensorType.Shape);
            strides = GetStrides(dims);
        }
        else
        {
            var dividedType = DistributedUtility.GetDividedTensorType(distributedType);
            dims = CompilerServices.GetMaxShape(dividedType.Shape);
            strides = GetStrides(dims);
        }

        var maxSize = GetProduct(dims) * tensorType.DType.SizeInBytes;
        return (maxSize, strides);
    }

    public static (Dimension MaxSize, Dimension[] Strides) GetTensorMaxSizeAndStridesExpr(TensorType tensorType, DistributedType? distributedType)
    {
        var (maxSize, strides) = GetTensorMaxSizeAndStrides(tensorType, distributedType);
        return (maxSize, strides.Select(x => (Dimension)x).ToArray());
    }

    public static (Dimension Size, Dimension[] Strides) GetTensorSizeAndContiguousStrides(TensorType tensorType, DistributedType? distributedType)
    {
        Dimension[] dims;
        Dimension[] strides;
        if (distributedType is null)
        {
            dims = ((RankedShape)tensorType.Shape).Dimensions.ToArray();
            strides = GetStrides(dims);
        }
        else
        {
            var dividedType = DistributedUtility.GetDividedTensorType(distributedType);
            dims = ((RankedShape)dividedType.Shape).Dimensions.ToArray();
            strides = GetStrides(dims);
        }

        var size = (Dimension)GetProduct(dims) * tensorType.DType.SizeInBytes;
        return (size, strides);
    }

    public static (long MaxSize, long[] Strides) GetTensorMaxSizeAndStrides(IRType type)
        => type switch
        {
            TensorType tensorType => GetTensorMaxSizeAndStrides(tensorType, null),
            DistributedType distributedType => GetTensorMaxSizeAndStrides(DistributedUtility.GetDividedTensorType(distributedType), distributedType),
            _ => throw new NotSupportedException(),
        };
}
