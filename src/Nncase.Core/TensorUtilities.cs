// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

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
    public static long GetProduct(ReadOnlySpan<int> dimensions, int startIndex = 0)
    {
        if (dimensions.Length == 0)
        {
            return 1;
        }

        long product = 1;
        for (int i = startIndex; i < dimensions.Length; i++)
        {
            if (dimensions[i] < 0)
            {
                throw new ArgumentOutOfRangeException($"{nameof(dimensions)}[{i}]");
            }

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
    /// Get the Expr Product.
    /// </summary>
    public static Expr GetProduct(ReadOnlySpan<Expr> dimensions, int startIndex = 0)
    {
        if (dimensions.Length == 0)
        {
            return 1;
        }

        Expr product = 1;
        for (int i = startIndex; i < dimensions.Length; i++)
        {
            var dimension = dimensions[i];
            product *= IR.F.Math.Require(dimension >= 0, dimension, "Dimension is out of range.");
        }

        return product;
    }

    public static bool IsAscending(ReadOnlySpan<int> values)
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

    public static bool IsDescending(ReadOnlySpan<int> values)
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
    public static int[] GetStrides(ReadOnlySpan<int> dimensions, bool reverseStride = false)
    {
        if (dimensions.IsEmpty)
        {
            return Array.Empty<int>();
        }

        int[] strides = new int[dimensions.Length];

        int stride = 1;
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
    /// get strides.
    /// </summary>
    public static Expr[] GetStrides(ReadOnlySpan<Expr> dimensions, bool reverseStride = false)
    {
        if (dimensions.IsEmpty)
        {
            return Array.Empty<Expr>();
        }

        var strides = new Expr[dimensions.Length];

        Expr stride = 1;
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

    public static void SplitStrides(int[] strides, int[] splitAxes, int[] newStrides, int stridesOffset, int[] splitStrides, int splitStridesOffset)
    {
        int newStrideIndex = 0;
        for (int i = 0; i < strides.Length; i++)
        {
            int stride = strides[i];
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
    public static int GetIndex(ReadOnlySpan<int> strides, ReadOnlySpan<int> indices, int startFromDimension = 0)
    {
        // Scalar
        if (strides.Length == 0)
        {
            if (indices.Length != 1 || indices[0] != 0)
            {
                throw new ArgumentOutOfRangeException(nameof(indices));
            }

            return 0;
        }

        Trace.Assert(strides.Length == indices.Length);

        int index = 0;
        for (int i = startFromDimension; i < indices.Length; i++)
        {
            index += strides[i] * indices[i];
        }

        return index;
    }

    /// <summary>
    /// get index.
    /// </summary>
    public static IR.Expr GetIndex(ReadOnlySpan<IR.Expr> strides, ReadOnlySpan<IR.Expr> indices, int startFromDimension = 0)
    {
        // Scalar
        if (strides.Length == 0)
        {
            if (indices.Length != 1)
            {
                throw new ArgumentOutOfRangeException(nameof(indices));
            }

            return IR.F.Math.Require(IR.F.Math.Equal(indices[0], 0), 0);
        }

        Trace.Assert(strides.Length == indices.Length);

        IR.Expr index = 0;
        for (int i = startFromDimension; i < indices.Length; i++)
        {
            index += strides[i] * indices[i];
        }

        return index;
    }

    /// <summary>
    /// Calculates the n-d indices from the 1-d index in a layout specificed by strides.
    /// </summary>
    public static void GetIndices(ReadOnlySpan<int> strides, bool reverseStride, int index, int[] indices, int startFromDimension = 0)
    {
        Trace.Assert(reverseStride ? IsAscending(strides) : IsDescending(strides), "Index decomposition requires ordered strides");
        Trace.Assert(strides.Length == indices.Length);

        int remainder = index;
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
    public static void GetIndices(ReadOnlySpan<int> strides, bool reverseStride, int index, Span<int> indices, int startFromDimension = 0)
    {
        Trace.Assert(reverseStride ? IsAscending(strides) : IsDescending(strides), "Index decomposition requires ordered strides");
        Trace.Assert(strides.Length == indices.Length);

        int remainder = index;
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
    public static int TransformIndexByStrides(int index, int[] sourceStrides, bool sourceReverseStride, int[] transformStrides)
    {
        Trace.Assert(index >= 0);
        Trace.Assert(sourceReverseStride ? IsAscending(sourceStrides) : IsDescending(sourceStrides), "Index decomposition requires ordered strides");
        Trace.Assert(sourceStrides.Length == transformStrides.Length);

        int transformIndex = 0;
        int remainder = index;

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
    public static bool IsContiguous(ReadOnlySpan<int> dimensions, ReadOnlySpan<int> strides)
    {
        return System.Collections.StructuralComparisons.StructuralEqualityComparer.Equals(GetStrides(dimensions), strides.ToArray());
    }

    /// <summary>
    /// check the dimensions selected range is contiguous.
    /// </summary>
    public static bool IsContiguousSlice(ReadOnlySpan<int> dimensions, ReadOnlySpan<System.Range> slices)
    {
        if (dimensions.Length != slices.Length)
        {
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
                int x when x == dimensions[i] => status switch
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
                int x when x > 0 && x < dimensions[i] => status switch
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
                return false;
            }
        }

        return true;
    }

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
            ints[i] = (int)longs[i];
        }

        return ints;
    }

    public static int[] ToInts(this long[] longs) => ToInts((ReadOnlySpan<long>)longs);

    public static int GetSize(Span<int> shapes, Span<int> strides, int elementSize)
    {
        int size = 0;
        for (int i = 0; i < shapes.Length; i++)
        {
            size += (shapes[i] - 1) * strides[i];
        }

        size += 1;
        return size * elementSize;
    }
}
