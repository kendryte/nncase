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

        var start = (int)TensorUtilities.GetIndex(Strides, starts);
        var size = (int)TensorUtilities.GetSize(shape, Strides, 1);
        size = Math.Min(size, Buffer.Length - start);
        var subBuffer = Buffer.Slice(start, size);
        return new Tensor<T>(subBuffer, shape, Strides);
    }

    public override Tensor<T> Squeeze(params int[] axes)
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

    public override Tensor<T> AsContiguous()
    {
        if (TensorUtilities.IsContiguous(Dimensions, Strides))
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
            throw new ArgumentException("the dest tensor shape must be equal to this tensor shape.", "dest");
        }

        var conti_dims = Math.Min(
            TensorUtilities.GetContiguousDims(src.Dimensions, src.Strides),
            TensorUtilities.GetContiguousDims(dest.Dimensions, dest.Strides));

        void Apply(int axis, long[] index)
        {
            if (axis >= (src.Rank - conti_dims))
            {
                var size = TensorUtilities.GetProduct(src.Dimensions, axis);

                var srcSpan = src.Buffer.Slice((int)TensorUtilities.GetIndex(src.Strides, index), (int)size);
                var destSpan = dest.Buffer.Slice((int)TensorUtilities.GetIndex(dest.Strides, index), (int)size);
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
