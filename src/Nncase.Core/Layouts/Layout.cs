// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Diagnostics;
using System.Linq;
using Nncase.IR;

namespace Nncase.Layouts;

public sealed record Layout : RecursiveValue, IEnumerable<Layout>
{
    public Layout(RecursiveValue shape, RecursiveValue? stride = null)
    {
        Shape = shape;
        Stride = stride ?? LayoutUtilities.PrefixProduct(shape);
    }

    public Layout(CollectValue shape, CollectValue? stride = null)
    {
        Shape = shape;
        Stride = stride ?? LayoutUtilities.PrefixProduct(shape);
    }

    public RecursiveValue Shape { get; }

    public RecursiveValue Stride { get; }

    public int Rank => Shape switch
    {
        CollectValue shape => shape.Elements.Length,
        IntValue => 1,
        _ => throw new InvalidOperationException("Invalid shape type"),
    };

    public Layout this[int i]
    {
        get
        {
            if (Shape is CollectValue shapeTuple && Stride is CollectValue strideTuple)
            {
                return new Layout(shapeTuple.Elements[i], strideTuple.Elements[i]);
            }
            else
            {
                Trace.Assert(i == 0);
                return new Layout(Shape, Stride);
            }
        }
    }

    public Layout this[Range range]
    {
        get
        {
            var (start, end) = (range.Start.GetOffset(Rank), range.End.GetOffset(Rank));
            return LayoutUtilities.MakeLayout(Enumerable.Range(start, end - start).Select(i => this[i]).ToArray());
        }
    }

    /// <summary>
    /// Create layout from tensor type.
    /// the stride is calculated by suffix product.
    /// </summary>
    /// <param name="tensorType">The tensor type.</param>
    /// <param name="bytes">Whether to consider byte size.</param>
    /// <returns>layout.</returns>
    public static Layout From(TensorType tensorType, bool bytes = false)
    {
        var shape = new CollectValue(CompilerServices.GetMaxShape(tensorType.Shape).Select(i => (IntValue)i).ToArray());
        var stride = LayoutUtilities.SuffixProduct(shape, bytes ? tensorType.DType.SizeInBytes : 1);
        return new Layout(shape, stride);
    }

    /// <summary>
    /// Create layout from distributed type. layout((times, shard_shape):(times_stride, shard_stride))
    /// It will add the outer dimension according to the split sharding.
    /// </summary>
    /// <param name="distType">The distributed type.</param>
    /// <param name="bytes">Whether to consider byte size.</param>
    /// <returns>layout.</returns>
    public static Layout From(DistributedType distType, bool bytes = false)
    {
        var layout = From(distType.TensorType, bytes);
        var tiler = LayoutUtilities.GetTiler(layout.Shape, distType.AxisPolicies, distType.Placement);
        var shard = LayoutUtilities.ZippedDivide(layout, tiler);
        var filtered = LayoutUtilities.Filter(shard[0], 1);
        return LayoutUtilities.MakeLayout(LayoutUtilities.Unflatten(filtered, new CollectValue(Enumerable.Range(0, filtered.Rank).Select(i => new IntValue(i)))), shard[1]);
    }

    /// <summary>
    /// the csharp not support same name method with different return type.
    /// so we split __call__ method to Slice and Offset.
    /// </summary>
    public Layout Slice(RecursiveValue coord)
    {
        return new Layout(LayoutUtilities.Slice(coord, Shape), LayoutUtilities.Slice(coord, Stride));
    }

    public long Invoke(RecursiveValue coord)
    {
        return LayoutUtilities.Crd2Idx(coord, Shape, Stride);
    }

    public RecursiveValue HierCoord(long idx)
    {
        return LayoutUtilities.Idx2Crd(idx, Shape, Stride);
    }

    public long Size() => LayoutUtilities.Product(Shape);

    public long Cosize() => Invoke(new IntValue(Size() - 1)) + 1;

    public override string ToString() => $"Layout({Shape}:{Stride})";

    public IEnumerator<Layout> GetEnumerator() => Enumerable.Range(0, Rank).Select(i => this[i]).GetEnumerator();

    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
}
