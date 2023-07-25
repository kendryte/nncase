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
public abstract class Buffer : Expr
{
    public Buffer(string name, DataType elemType, MemoryLocation memoryLocation, Expr[] operands)
        : base(operands.ToArray())
    {
        Name = name;
        ElemType = elemType;
        MemLocation = memoryLocation;
    }

    public string Name { get; }

    public DataType ElemType { get; }

    public MemoryLocation MemLocation { get; }

    /// <summary>
    /// Gets if this buffer from the constant !.
    /// </summary>
    public TensorConst? Const { get; init; }

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
    public abstract ReadOnlySpan<Expr> Strides { get; }

    /// <summary>
    /// Gets the shape.
    /// </summary>
    public abstract ReadOnlySpan<Expr> Dimensions { get; }

    /// <inheritdoc/>
    public override bool Equals(object? obj)
    {
        if (obj is not Buffer other)
        {
            return false;
        }

        if (Const is not null && !Const.Equals(other.Const))
        {
            return false;
        }

        return string.Equals(Name, other.Name, StringComparison.Ordinal) &&
                ElemType.Equals(other.ElemType) &&
                MemLocation.Equals(other.MemLocation) &&
                Rank.Equals(other.Rank) &&
                base.Equals(obj);
    }
}

/// <summary>
/// the logical buffer.
/// </summary>
public sealed class LogicalBuffer : Buffer
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
    public LogicalBuffer(string name, DataType elemType, MemoryLocation location, ReadOnlySpan<Expr> dimensions, ReadOnlySpan<Expr> strides)
        : base(name, elemType, location, ArrayUtility.Concat(dimensions, strides))
    {
        Rank = dimensions.Length;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="LogicalBuffer"/> class.
    /// <see cref="LogicalBuffer"/>.
    /// </summary>
    public LogicalBuffer(string name, MemoryLocation location, TensorConst tensor)
        : this(name, tensor.Value.ElementType, location, ArrayUtility.ToExprArray(tensor.Value.Dimensions), ArrayUtility.ToExprArray(tensor.Value.Strides))
    {
        Const = tensor;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="LogicalBuffer"/> class.
    /// <seealso cref="LogicalBuffer"/>
    /// </summary>
    public LogicalBuffer(string name, DataType elemType, MemoryLocation location, ReadOnlySpan<Expr> dimensions)
        : this(name, elemType, location, dimensions, TensorUtilities.GetStrides(dimensions))
    {
    }

    /// <summary>
    /// Gets get the total length.
    /// </summary>
    public Expr Length => TensorUtilities.GetProduct(Dimensions);

    /// <summary>
    /// Gets the shape.
    /// </summary>
    public override ReadOnlySpan<Expr> Dimensions => Operands[0..Rank];

    /// <summary>
    /// Gets the strides.
    /// </summary>
    public override ReadOnlySpan<Expr> Strides => Operands[Rank..];

    /// <inheritdoc/>
    public override int Rank { get; }

    /// <inheritdoc/>
    public override string ToString()
    {
        return $"LogicalBuffer({Name}, {ElemType}, {nameof(MemLocation)})";
    }

    /// <inheritdoc/>
    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context)
        => functor.VisitLogicalBuffer(this, context);

    public LogicalBuffer With(string? name = null, DataType? elemType = null, MemoryLocation? location = null, Expr[]? dimensions = null, Expr[]? strides = null)
        => new LogicalBuffer(name ?? Name, elemType ?? ElemType, location ?? MemLocation, dimensions ?? Dimensions, strides ?? Strides) { Const = Const };
}

/// <summary>
/// the physical buffer.
/// </summary>
public sealed class PhysicalBuffer : Buffer
{
    private readonly int[] _fixedDimensions;
    private readonly int[] _fixedStrides;

    /// <summary>
    /// Initializes a new instance of the <see cref="PhysicalBuffer"/> class.
    /// ctor for physical buffer.
    /// </summary>
    public PhysicalBuffer(string name, DataType elemType, MemoryLocation location, ReadOnlySpan<int> dimensions, ReadOnlySpan<int> strides, int start, int size)
        : base(name, elemType, location, Array.Empty<Expr>())
    {
        Start = start;
        Size = size;
        _fixedDimensions = dimensions.ToArray();
        _fixedStrides = strides.ToArray();
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="PhysicalBuffer"/> class.
    /// <see cref="PhysicalBuffer"/>.
    /// </summary>
    public PhysicalBuffer(string name, DataType elemType, MemoryLocation location, ReadOnlySpan<int> dimensions, int start, int size)
        : this(name, elemType, location, dimensions, TensorUtilities.GetStrides(dimensions), start, size)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="PhysicalBuffer"/> class.
    /// <see cref="PhysicalBuffer"/>.
    /// </summary>
    public PhysicalBuffer(string name, MemoryLocation location, TensorConst tensor, int start, int size)
        : this(name, tensor.Value.ElementType, location, tensor.Value.Dimensions, tensor.Value.Strides, start, size)
    {
        Const = tensor;
    }

    /// <summary>
    /// Gets fixed dimensions.
    /// </summary>
    public ReadOnlySpan<int> FixedDimensions => _fixedDimensions;

    /// <summary>
    /// Gets fixed strides.
    /// </summary>
    public ReadOnlySpan<int> FixedStrides => _fixedStrides;

    /// <summary>
    /// Gets or sets start.
    /// </summary>
    public int Start { get; set; }

    /// <summary>
    /// Gets total size in bytes.
    /// </summary>
    public int Size { get; init; }

    /// <summary>
    /// Gets dimensions.
    /// </summary>
    public override ReadOnlySpan<Expr> Dimensions => ArrayUtility.ToExprArray(FixedDimensions);

    /// <summary>
    /// Gets strides.
    /// </summary>
    public override ReadOnlySpan<Expr> Strides => ArrayUtility.ToExprArray(FixedStrides);

    /// <summary>
    /// Gets shape.
    /// </summary>
    public Shape Shape => new Shape(FixedDimensions);

    /// <inheritdoc/>
    public override int Rank => FixedDimensions.Length;

    /// <inheritdoc/>
    public override string ToString()
    {
        return $"PhysicalBuffer({Name}, {ElemType}, {nameof(MemLocation)})";
    }

    /// <inheritdoc/>
    public override bool Equals(object? obj)
    {
        return base.Equals(obj) && obj is PhysicalBuffer other &&
          FixedDimensions.SequenceEqual(other.FixedDimensions) &&
          FixedStrides.SequenceEqual(other.FixedStrides);
    }

    /// <inheritdoc/>
    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context)
        => functor.VisitPhysicalBuffer(this, context);

    public PhysicalBuffer With(string? name = null, DataType? elemType = null, MemoryLocation? location = null, int[]? dimensions = null, int[]? strides = null, int? start = null, int? size = null)
        => new PhysicalBuffer(name ?? Name, elemType ?? ElemType, location ?? MemLocation, dimensions ?? FixedDimensions, strides ?? FixedStrides, start ?? Start, size ?? Size) { Const = Const };
}
