// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase;

public partial class Tensor<T>
{
    /// <summary>
    /// Cast and copy to array.
    /// </summary>
    /// <returns>Casted array.</returns>
    public T[] ToArray()
    {
        if (IsContiguous)
        {
            var array = new T[Length];
            Buffer.CopyTo(array);
            return array;
        }
        else
        {
            var array = new T[TensorUtilities.GetProduct(Dimensions)];
            CopyTo(new Tensor<T>(array, Dimensions));
            return array;
        }
    }

    /// <summary>
    /// Cast to a scalar.
    /// </summary>
    /// <returns>Casted scalar.</returns>
    public T ToScalar()
    {
        if (Length != 1)
        {
            throw new InvalidOperationException("This tensor is not a scalar.");
        }

        return Buffer.Span[0];
    }

    /// <summary>
    /// Create the view from a <see cref="Tensor{T}"/>.
    /// </summary>
    /// <returns> tensor view. </returns>
    public override Tensor<T> View(ReadOnlySpan<long> starts, ReadOnlySpan<long> shape)
    {
        if (starts.Length != shape.Length || starts.Length != Dimensions.Length)
        {
            throw new ArgumentOutOfRangeException("starts", "the starts and shape must be equal to this tensor rank.");
        }

        var start = (int)TensorUtilities.GetLinearOffset(Strides, starts);
        var size = (int)TensorUtilities.GetSize(shape, Strides, 1);
        size = Math.Min(size, Buffer.Length - start);
        var subBuffer = Buffer.Slice(start, size);
        return new Tensor<T>(subBuffer, shape, Strides);
    }

    public override Tensor Transpose(ReadOnlySpan<long> perm)
    {
        if (perm.Length != Rank || perm.ToArray().Any(x => x >= Rank))
        {
            throw new ArgumentException("Permutation length must match tensor rank", nameof(perm));
        }

        var invPerms = perm.ToInts().Zip(Enumerable.Range(0, Rank)).OrderBy(p => p.First).Select(p => p.Second).ToArray();
        var permArr = perm.ToInts();
        var destDimensions = Enumerable.Range(0, Rank).Select(i => Dimensions[permArr[i]]).ToArray();
        var destStrides = TensorUtilities.GetDefaultStrides(destDimensions);
        void Apply(int axis, Span<int> index, int i, int j, Span<T> srcSpan, Span<T> destSpan)
        {
            if (axis < Rank - 1)
            {
                for (index[axis] = 0; index[axis] < Dimensions[axis]; index[axis]++)
                {
                    int ni = i + (index[axis] * (int)Strides[axis]);
                    int nj = j + (index[axis] * (int)destStrides[invPerms[axis]]);
                    Apply(axis + 1, index, ni, nj, srcSpan, destSpan);
                }
            }
            else
            {
                for (index[axis] = 0; index[axis] < Dimensions[axis]; index[axis]++)
                {
                    int ni = i + (index[axis] * (int)Strides[axis]);
                    int nj = j + (index[axis] * (int)destStrides[invPerms[axis]]);
                    destSpan[nj] = srcSpan[ni];
                }
            }
        }

        // 0,  4,  8, 12, 16, 20,  1,  5,  9, 13, 17, 21,  2,  6, 10, 14, 18, 22,  3,  7, 11, 15, 19, 23
        var newBuffer = new T[Buffer.Length];
        var newBufferMemory = new Memory<T>(newBuffer);
        var bufferSpan = Buffer.Span;
        var indices = Enumerable.Repeat(0, Rank).ToArray();
        Apply(0, indices, 0, 0, bufferSpan, newBufferMemory.Span);
        return new Tensor<T>(newBuffer, destDimensions, destStrides);
    }

    public override Tensor<T> Squeeze(params long[] axes)
    {
        var dimensions = Enumerable.Range(0, Rank).Where(i =>
        {
            if (axes.Contains(i))
            {
                if (Dimensions[i] != 1)
                {
                    throw new ArgumentOutOfRangeException("axes", "the axes dimension must be 1.");
                }

                return false;
            }

            return true;
        }).Select(i => Dimensions[i]).ToArray();
        var strides = Enumerable.Range(0, Rank).Where(i => !axes.Contains(i)).Select(i => Strides[i]).ToArray();
        return new Tensor<T>(Buffer, dimensions, strides);
    }

    public override void CopyTo(Tensor dest)
    {
        CopyTo(this, dest.Cast<T>());
    }

    public override Tensor<T> AsContiguous(bool force = false)
    {
        if (!force && TensorUtilities.IsContiguous(Dimensions, Strides))
        {
            return this;
        }

        var dest = Tensor.Zeros(ElementType, Dimensions).Cast<T>();
        CopyTo(dest);
        return dest;
    }

    private static void CopyTo(Tensor<T> src, Tensor<T> dest)
    {
        if (!src.Dimensions.SequenceEqual(dest.Dimensions))
        {
            throw new ArgumentException($"the dest tensor shape [{string.Join(" ", dest.Dimensions.ToArray())}] must be equal to this tensor shape [{string.Join(" ", src.Dimensions.ToArray())}].", "dest");
        }

        var conti_dims = Math.Min(
            TensorUtilities.GetContiguousDims(src.Dimensions, src.Strides),
            TensorUtilities.GetContiguousDims(dest.Dimensions, dest.Strides));

        void Apply(int axis, long[] index)
        {
            if (axis >= (src.Rank - conti_dims))
            {
                var size = TensorUtilities.GetProduct(src.Dimensions, axis);

                var srcSpan = src.Buffer.Slice((int)TensorUtilities.GetLinearOffset(src.Strides, index), (int)size);
                var destSpan = dest.Buffer.Slice((int)TensorUtilities.GetLinearOffset(dest.Strides, index), (int)size);
                srcSpan.CopyTo(destSpan);
            }
            else
            {
                var dim = src.Dimensions[axis];
                for (index[axis] = 0; index[axis] < dim; index[axis]++)
                {
                    Apply(axis + 1, index);
                }
            }
        }

        long[] index = new long[src.Dimensions.Length];
        Apply(0, index);
    }
}
