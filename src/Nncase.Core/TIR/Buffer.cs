// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using Nncase.IR;
using Nncase.Utilities;

namespace Nncase.TIR;

/// <summary>
/// the buffer view interface.
/// </summary>
public interface IBufferView<T>
  where T : class
{
    /// <summary>
    /// Gets the parent.
    /// </summary>
    public T Parent { get; init; }

    /// <summary>
    /// Gets the root parent.
    /// </summary>
    public T RootParent { get; init; }

    /// <summary>
    /// Gets the select slice ranges.
    /// </summary>
    public ReadOnlySpan<SelectedRange> SelectedRanges { get; }

    /// <summary>
    /// Gets get current stride.
    /// </summary>
    public ReadOnlySpan<int> Stride { get; }

    /// <summary>
    /// Gets the shape of this buffer view.
    /// </summary>
    public ReadOnlySpan<int> Dimensions { get; }

    /// <summary>
    /// Gets get the DType.
    /// </summary>
    public DataType DType { get; init; }

    /// <summary>
    /// Gets a value indicating whether check if the buffer is sliced.
    /// </summary>
    public bool IsSubView { get; init; }

    /// <summary>
    /// support slice like the normal array.
    /// </summary>
    /// <param name="segments">the slice info.</param>
    /// <returns>self sub buffer.</returns>
    public T this[SegmentND segments] { get; }

    /// <summary>
    /// support slice like the normal array.
    /// </summary>
    /// <param name="segments">the slice info.</param>
    /// <returns> self sub buffer. </returns>
    public T this[params Segment1D[] segments] { get; }
}

/// <summary>
/// the padding.
/// </summary>
public record Padding(int Before, int After, int Interior = 0)
{
    /// <summary>
    /// get left right padding sum.
    /// </summary>
    public int Sum()
    {
        return Before + After;
    }

    /// <summary>
    /// zero pad.
    /// </summary>
    public static Padding Zero()
    {
        return new(0, 0, 0);
    }
}

public record Segment1D
{
    public Segment1D(System.Range range, Padding padding, int index = 0)
    {
        if (range.Start.IsFromEnd)
        {
            throw new NotSupportedException("The Negative Start Slice");
        }

        if (range.End.IsFromEnd && !range.Equals(System.Range.All))
        {
            throw new NotSupportedException("The Negative Slice For The Tensor.");
        }

        Range = range;
        Padding = padding;
        Index = index;
    }

    public System.Range Range { get; set; }

    public Padding Padding { get; set; }

    public int Index { get; set; }

    public int Start => Range.Start.Value;

    public int End => Range.End.Value;

    public int Length
    {
        get
        {
            if (Range.Equals(System.Range.All))
            {
                throw new InvalidOperationException("Range.Equals(Range.All)");
            }

            return Range.End.Value - Range.Start.Value;
        }
    }

    public static implicit operator Segment1D(System.Range range)
    {
        return new(range, Padding.Zero());
    }

    public static Segment1D operator /(Segment1D seg, int scale)
    {
        if (seg.Range.Equals(System.Range.All))
        {
            throw new ArgumentOutOfRangeException(nameof(seg), "The All Slice Can't Be Divide!");
        }

        return new(new(seg.Range.Start.Value / scale, seg.Range.End.Value / scale), seg.Padding);
    }

    public static Segment1D operator *(Segment1D seg, int scale)
    {
        return new(new(seg.Range.Start.Value * scale, seg.Range.End.Value * scale), seg.Padding);
    }

    public static Segment1D operator +(Segment1D lhs, Segment1D rhs)
    {
        var min_start = Math.Min(lhs.Start, rhs.Start);
        var max_end = Math.Max(lhs.End, rhs.End);
        return new Segment1D(min_start..max_end, Padding.Zero());
    }

    public override string ToString()
    {
        return $"{Range}";
    }
}

public class SegmentND : IEnumerable<Segment1D>, IReadOnlyList<Segment1D>
{
    private readonly Segment1D[] _segments;

    public SegmentND(IEnumerable<Segment1D> segments)
        : this(segments.ToArray())
    {
    }

    public SegmentND(ReadOnlySpan<Segment1D> segments)
    {
        _segments = new Segment1D[segments.Length];
        segments.CopyTo(_segments);
    }

    public SegmentND(params Segment1D[] segments)
        : this(segments.AsSpan())
    {
    }

    public Padding PadH => _segments[2].Padding;

    public Padding PadW => _segments[3].Padding;

    public ReadOnlySpan<Segment1D> Segments => _segments;

    /// <summary>
    /// Gets todo remove it.
    /// </summary>
    public int Shape_size => _segments.Aggregate(1, (acc, seg) => acc * seg.Length);

    public int Count => ((IReadOnlyCollection<Segment1D>)_segments).Count;

    public Segment1D this[int index]
    {
        get => _segments[index];
        set => _segments[index] = value;
    }

    public static bool operator ==(SegmentND lhs, SegmentND rhs)
    {
        return lhs.Equals(rhs);
    }

    public static bool operator !=(SegmentND lhs, SegmentND rhs)
    {
        return !(lhs == rhs);
    }

    public static SegmentND operator +(SegmentND lhs, SegmentND rhs)
    {
        return new(lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2], lhs[3] + rhs[3]);
    }

    public override bool Equals(object? obj)
    {
        return obj is SegmentND segment &&
               StructuralComparisons.StructuralEqualityComparer.Equals(_segments, segment._segments);
    }

    public override int GetHashCode()
    {
        return HashCode.Combine(StructuralComparisons.StructuralEqualityComparer.GetHashCode(_segments), PadH, PadW);
    }

    public IEnumerator<Segment1D> GetEnumerator()
    {
        return ((IEnumerable<Segment1D>)_segments).GetEnumerator();
    }

    IEnumerator IEnumerable.GetEnumerator()
    {
        return _segments.GetEnumerator();
    }

    /// <inheritdoc/>
    public override string ToString()
    {
        return $"{string.Join(",", _segments.Select(s => s.ToString()))}";
    }
}

public record SelectedRange(int Start, int End, Padding Padding)
{
    public SelectedRange Slice(Segment1D segment)
    {
        if (segment.Range.Equals(System.Range.All))
        {
            return this with { };
        }

        if (!(segment.Start >= Start && segment.End <= End))
        {
            throw new NotSupportedException("!(segment.Start >= Start && segment.End <= End)");
        }

        return new(segment.Start, segment.End, segment.Padding);
    }
}

/// <summary>
/// buffer.
/// </summary>
public sealed class Buffer : Expr
{
    private static int _globalVarIndex;

    public Buffer(string name, DataType elemType, MemSpan memSpan, Expr[] dimensions, Expr[] strides)
        : base(new[] { memSpan }.Concat(dimensions).Concat(strides))
    {
        Name = name;
        ElemType = elemType;
        Rank = dimensions.Length;
        GlobalVarIndex = Interlocked.Increment(ref _globalVarIndex);
    }

    public string Name { get; }

    public DataType ElemType { get; }

    /// <summary>
    /// Gets rank of the tensor: number of dimensions.
    /// </summary>
    public int Rank { get; }

    /// <summary>
    /// Gets the global var index.
    /// </summary>
    public int GlobalVarIndex { get; }

    /// <summary>
    /// Gets the shape.
    /// </summary>
    public MemSpan MemSpan => (MemSpan)Operands[0];

    /// <summary>
    /// Gets the shape.
    /// </summary>
    public ReadOnlySpan<Expr> Dimensions => Operands[1..(1 + Rank)];

    /// <summary>
    /// Gets the strides.
    /// <remarks>
    /// This Strides is by elements not by bytes!
    /// </remarks>
    /// </summary>
    public ReadOnlySpan<Expr> Strides => Operands[(1 + Rank)..(1 + Rank + Rank)];

    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context) => throw new NotImplementedException();
}