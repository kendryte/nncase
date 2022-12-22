// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using Nncase.IR;

namespace Nncase.TIR;

/// <summary>
/// the padding.
/// </summary>
/// <param name="Before"></param>
/// <param name="After"></param>
/// <param name="Interior"></param>
public record Padding(int Before, int After, int Interior = 0)
{
    /// <summary>
    /// get left right padding sum.
    /// </summary>
    /// <returns></returns>
    public int Sum()
    {
        return Before + After;
    }

    /// <summary>
    /// zero pad.
    /// </summary>
    /// <returns></returns>
    public static Padding Zero()
    {
        return new(0, 0, 0);
    }
}

public record Segment1D
{
    public System.Range Range;
    public Padding Padding;

    public int Index;

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
            throw new ArgumentOutOfRangeException("The All Slice Can't Be Divide!");
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
/// the buffer view interface.
/// </summary>
/// <typeparam name="T"></typeparam>
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
/// buffer.
/// </summary>
/// <param name="Name"></param>
/// <param name="ElemType"></param>
/// <param name="MemLocation"></param>
public abstract record Buffer(string Name, DataType ElemType, Schedule.MemoryLocation MemLocation) : Expr
{
    /// <summary>
    /// if this buffer from the constant !.
    /// </summary>
    public TensorConst? Const;

    /// <summary>
    /// Gets rank of the tensor: number of dimensions.
    /// </summary>
    public abstract int Rank { get; }

    /// <summary>
    /// Gets the strides.
    /// <remarks>
    /// This Strides is by elements not by bytes!
    /// </remarks>
    /// </summary>
    public abstract IRArray<Expr> Strides { get; }

    /// <summary>
    /// Gets the shape.
    /// </summary>
    public abstract IRArray<Expr> Dimensions { get; }
}

/// <summary>
/// the logical buffer.
/// </summary>
/// <param name="Name"></param>
/// <param name="ElemType"></param>
/// <param name="MemLocation"></param>
public sealed record LogicalBuffer(string Name, DataType ElemType, Schedule.MemoryLocation MemLocation) : Buffer(Name, ElemType, MemLocation)
{
    /// <summary>
    /// Initializes a new instance of the <see cref="LogicalBuffer"/> class.
    /// create from the IRType.
    /// </summary>
    /// <param name="name">the name.</param>
    /// <param name="location">the location.</param>
    /// <param name="elemType">prim type.</param>
    /// <param name="dimensions">the shape.</param>
    /// <param name="strides">the strides.</param>
    public LogicalBuffer(string name, DataType elemType, Schedule.MemoryLocation location, IRArray<Expr> dimensions, IRArray<Expr> strides)
        : this(name, elemType, location)
    {
        Dimensions = dimensions;
        Strides = strides;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="LogicalBuffer"/> class.
    /// <see cref="LogicalBuffer"/>.
    /// </summary>
    /// <param name="name"></param>
    /// <param name="location"></param>
    /// <param name="tensor"></param>
    public LogicalBuffer(string name, Schedule.MemoryLocation location, TensorConst tensor)
        : this(name, tensor.Value.ElementType, location,
     ImmutableArray.Create<Expr>(tensor.Value.Dimensions.ToArray()), ImmutableArray.Create<Expr>(tensor.Value.Strides.ToArray()))
    {
        Const = tensor;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="LogicalBuffer"/> class.
    /// <seealso cref="LogicalBuffer"/>
    /// </summary>
    /// <param name="name"></param>
    /// <param name="location"></param>
    /// <param name="elemType"></param>
    /// <param name="dimensions"></param>
    public LogicalBuffer(string name, DataType elemType, Schedule.MemoryLocation location, IRArray<Expr> dimensions)
        : this(name, elemType, location, dimensions, TensorUtilities.GetStrides(dimensions).ToImmutableArray())
    {
    }

    /// <summary>
    /// Gets get the total length.
    /// </summary>
    public Expr Length => TensorUtilities.GetProduct(Dimensions);

    /// <summary>
    /// Gets the strides.
    /// </summary>
    public override IRArray<Expr> Strides { get; }

    /// <summary>
    /// Gets the shape.
    /// </summary>
    public override IRArray<Expr> Dimensions { get; }

    /// <inheritdoc/>
    public override int Rank => Dimensions.Count;

    /// <inheritdoc/>
    public override string ToString()
    {
        return $"LogicalBuffer({Name}, {ElemType}, {nameof(MemLocation)})";
    }

    /// <inheritdoc/>
    protected override bool PrintMembers(StringBuilder builder)
    {
        builder.Append($"LogicalBuffer({Name}, {ElemType}, {nameof(MemLocation)})");
        return true;
    }
}

/// <summary>
/// the physicall buffer.
/// </summary>
/// <param name="Name"></param>
/// <param name="ElemType"></param>
/// <param name="MemLocation"></param>
public sealed record PhysicalBuffer(string Name, DataType ElemType, Schedule.MemoryLocation MemLocation) : Buffer(Name, ElemType, MemLocation)
{
    /// <summary>
    /// get fixed dimensions.
    /// </summary>
    public int[] FixedDimensions = Array.Empty<int>();

    /// <summary>
    /// get fixed strides.
    /// </summary>
    public int[] FixedStrides = Array.Empty<int>();

    /// <summary>
    /// start.
    /// </summary>
    public int Start;

    /// <summary>
    /// total size in bytes.
    /// </summary>
    public int Size;

    /// <summary>
    /// Initializes a new instance of the <see cref="PhysicalBuffer"/> class.
    /// ctor for physical buffer.
    /// </summary>
    /// <param name="name"></param>
    /// <param name="location"></param>
    /// <param name="elemType"></param>
    /// <param name="dimensions"></param>
    /// <param name="stirdes"></param>
    /// <param name="start"></param>
    /// <param name="size"></param>
    public PhysicalBuffer(string name, DataType elemType, Schedule.MemoryLocation location, IEnumerable<int> dimensions, IEnumerable<int> stirdes, int start, int size)
        : this(name, elemType, location)
    {
        Start = start;
        Size = size;
        FixedDimensions = dimensions.ToArray();
        FixedStrides = stirdes.ToArray();
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="PhysicalBuffer"/> class.
    /// <see cref="PhysicalBuffer"/>.
    /// </summary>
    /// <param name="name"></param>
    /// <param name="elemType"></param>
    /// <param name="location"></param>
    /// <param name="dimensions"></param>
    /// <param name="start"></param>
    /// <param name="size"></param>
    public PhysicalBuffer(string name, DataType elemType, Schedule.MemoryLocation location, IEnumerable<int> dimensions, int start, int size)
        : this(name, elemType, location, dimensions, TensorUtilities.GetStrides(dimensions.ToArray()), start, size)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="PhysicalBuffer"/> class.
    /// <see cref="PhysicalBuffer"/>.
    /// </summary>
    /// <param name="name"></param>
    /// <param name="location"></param>
    /// <param name="tensor"></param>
    /// <param name="start"></param>
    /// <param name="size"></param>
    public PhysicalBuffer(string name, Schedule.MemoryLocation location, TensorConst tensor, int start, int size)
        : this(name, tensor.Value.ElementType, location, tensor.Value.Dimensions.ToArray(), tensor.Value.Strides.ToArray(), start, size)
    {
        Const = tensor;
    }

    /// <summary>
    /// Gets dimensions.
    /// </summary>
    public override IRArray<Expr> Dimensions => FixedDimensions.Length == 0 ?
      throw new ArgumentOutOfRangeException() :
      new(FixedDimensions.Select(i => (Expr)i));

    /// <summary>
    /// Gets strides.
    /// </summary>
    public override IRArray<Expr> Strides => FixedStrides.Length == 0 ?
      throw new ArgumentOutOfRangeException() :
      new(FixedStrides.Select(i => (Expr)i));

    /// <summary>
    /// Gets shape.
    /// </summary>
    public Shape Shape => new Shape(FixedDimensions);

    /// <inheritdoc/>
    public override int Rank => FixedDimensions.Rank;

    /// <inheritdoc/>
    public override string ToString()
    {
        return $"PhysicalBuffer({Name}, {ElemType}, {nameof(MemLocation)})";
    }

    /// <inheritdoc/>
    protected override bool PrintMembers(StringBuilder builder)
    {
        builder.Append($"PhysicalBuffer({Name}, {ElemType}, {nameof(MemLocation)})");
        return true;
    }
}
