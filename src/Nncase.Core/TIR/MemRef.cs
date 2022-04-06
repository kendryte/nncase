// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Nncase.IR;

namespace Nncase.TIR;


/// <summary>
/// the padding
/// </summary>
/// <param name="before"></param>
/// <param name="After"></param>
/// <param name="Interior"></param>
public record Padding(int before, int After, int Interior = 0)
{
    /// <summary>
    /// get left right padding sum.
    /// </summary>
    /// <returns></returns>
    public int sum() { return before + After; }

    /// <summary>
    /// zero pad.
    /// </summary>
    /// <returns></returns>
    public static Padding Zero() { return new(0, 0, 0); }
}

public record Segment1D
{
    public System.Range Range;
    public Padding Padding;
    public int Start => Range.Start.Value;
    public int End => Range.End.Value;
    public int Index;
    public int Length
    {
        get
        {
            if (Range.Equals(System.Range.All))
                throw new InvalidOperationException("Range.Equals(Range.All)");
            return Range.End.Value - Range.Start.Value;
        }
    }

    public Segment1D(System.Range range, Padding padding, int index = 0)
    {
        if (range.Start.IsFromEnd)
            throw new NotSupportedException("The Negative Start Slice");
        if (range.End.IsFromEnd && !range.Equals(System.Range.All))
            throw new NotSupportedException("The Negative Slice For The Tensor.");
        Range = range;
        Padding = padding;
        Index = index;
    }

    public static Segment1D operator /(Segment1D seg, int scale)
    {
        if (seg.Range.Equals(System.Range.All))
            throw new ArgumentOutOfRangeException("The All Slice Can't Be Divide!");
        return new(new(seg.Range.Start.Value / scale, seg.Range.End.Value / scale), seg.Padding);
    }

    public static Segment1D operator *(Segment1D seg, int scale)
    {
        return new(new(seg.Range.Start.Value * scale, seg.Range.End.Value * scale), seg.Padding);
    }

    public override string ToString()
    {
        return $"{Range}";
    }

    public static implicit operator Segment1D(System.Range range)
    {
        return new(range, Padding.Zero());
    }
}

public class SegmentND : IEnumerable<Segment1D>, IReadOnlyList<Segment1D>
{

    readonly Segment1D[] _segments;
    public Padding PadH => _segments[2].Padding;
    public Padding PadW => _segments[3].Padding;

    public ReadOnlySpan<Segment1D> Segments => _segments;

    public Segment1D this[int index]
    {
        get => _segments[index];
        set => _segments[index] = value;
    }

    public SegmentND(IEnumerable<Segment1D> segments) : this(segments.ToArray()) { }

    public SegmentND(ReadOnlySpan<Segment1D> segments)
    {
        _segments = new Segment1D[segments.Length];
        segments.CopyTo(_segments);
    }

    public SegmentND(params Segment1D[] segments) : this(segments.AsSpan())
    {
    }

    /// <summary>
    /// todo remove it
    /// </summary>
    public int shape_size => _segments.Aggregate(1, (acc, seg) => acc * seg.Length);

    public int Count => ((IReadOnlyCollection<Segment1D>)_segments).Count;

    public override bool Equals(object? obj)
    {
        return obj is SegmentND segment &&
               StructuralComparisons.StructuralEqualityComparer.Equals(_segments, segment._segments);
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
            return this with { };
        if (!(segment.Start >= Start && segment.End <= End))
            throw new NotSupportedException("!(segment.Start >= Start && segment.End <= End)");
        return new(segment.Start, segment.End, segment.Padding);
    }
}




/// <summary>
/// the buffer view interface
/// </summary>
/// <typeparam name="T"></typeparam>
public interface IBufferView<T>
  where T : class
{
    /// <summary>
    /// the parent.
    /// </summary>
    public T Parent { get; init; }

    /// <summary>
    /// the root parent.
    /// </summary>
    public T RootParent { get; init; }

    /// <summary>
    /// the select slice ranges.
    /// </summary>
    public ReadOnlySpan<SelectedRange> SelectedRanges { get; }

    /// <summary>
    /// get current stride
    /// </summary>
    public ReadOnlySpan<int> Stride { get; }

    /// <summary>
    /// the shape of this buffer view
    /// </summary>
    public ReadOnlySpan<int> Shape { get; }

    /// <summary>
    /// get the DType
    /// </summary>
    public DataType DType { get; init; }

    /// <summary>
    /// check if the buffer is sliced
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
/// todo tempory named memref, need change to buffer.
/// </summary>
public record Buffer : Expr, IBufferView<Buffer>, IStructuralEquatable
{

    /// <summary>
    /// if this buffer from the constant !
    /// </summary>
    public Const? Const;

    /// <summary>
    /// get the memref name.
    /// </summary>
    public string Name;

    /// <summary>
    /// 
    /// the memref pointed elem type
    /// </summary>
    public TensorType ElemType;

    /// <summary>
    /// get the memory loacation
    /// </summary>
    public Schedule.MemoryLocation MemLocation;

    /// <summary>
    /// set the cache level.
    /// </summary>
    public int CacheLevel;

    /// <summary>
    /// create from the IRType.
    /// </summary>
    /// <param name="name">the name.</param>
    /// <param name="location">the location.</param>
    /// <param name="elemType">prim type.</param>
    public Buffer(string name, Schedule.MemoryLocation location, TensorType elemType)
    {
        Name = name;
        ElemType = elemType;
        MemLocation = location;

        DType = ElemType switch
        {
            TensorType type => type.DType,
            _ => throw new NotSupportedException(ElemType.ToString()),
        };

        _shape = ((TensorType)elemType).Shape.ToValueArray();
        _stride = TensorUtilities.GetStrides(_shape).Select(s => s * DType.SizeInBytes).ToArray();
        _selectedRanges = _shape.Select(s => new SelectedRange(0, s, Padding.Zero())).ToArray();
        IsSubView = false;
        Parent = this;
        RootParent = this;
    }

    /// <summary>
    /// build new memref by segmentnd.
    /// </summary>
    /// <param name="segments"></param>
    /// <param name="parent"></param>
    /// <exception cref="InvalidOperationException"></exception>
    public Buffer(SegmentND segments, Buffer parent)
    {
        if (!(segments.Count == parent.SelectedRanges.Length))
            throw new InvalidOperationException("segments.Count == SelectedRanges.Length");
        _selectedRanges = segments.Zip(parent.SelectedRanges.ToArray()).Select(t => t.Item2.Slice(t.Item1)).ToArray();
        _shape = _selectedRanges.Select(s => s.End - s.Start).ToArray();
        _stride = parent.Stride.ToArray();
        IsSubView = true;
        Parent = parent;
        RootParent = parent.RootParent;

        Name = parent.Name;
        ElemType = parent.ElemType;
        DType = parent.DType;
        MemLocation = parent.MemLocation;
    }

    /// <inheritdoc/>
    public Buffer this[SegmentND segments] => new(segments, this);

    /// <inheritdoc/>
    public Buffer this[params Segment1D[] segments] => new(new(segments), this);

    /// <inheritdoc/>
    public Buffer Parent { get; init; }

    /// <inheritdoc/>
    public Buffer RootParent { get; init; }

    /// <inheritdoc/>
    public ReadOnlySpan<SelectedRange> SelectedRanges => _selectedRanges;

    /// <inheritdoc/>
    public ReadOnlySpan<int> Stride => _stride;

    /// <inheritdoc/>
    public ReadOnlySpan<int> Shape => _shape;

    /// <inheritdoc/>
    public DataType DType { get; init; }

    /// <inheritdoc/>
    public bool IsSubView { get; init; }


    private SelectedRange[] _selectedRanges;
    private int[] _stride;
    private int[] _shape;

    /// <summary>
    /// get the Addr expr .
    /// </summary>
    public Expr Addr => IR.F.Buffer.DDrOf(RootParent);

    /// <summary>
    /// get current Addr
    /// </summary>
    public Expr CurAddr => Addr + AddrOffset;

    /// <summary>
    /// get the allocate basement.
    /// </summary>
    public Expr BaseMent => IR.F.Buffer.BaseMentOf(RootParent);

    /// <summary>
    /// get current buffer view Addr Offset
    /// </summary>
    public int AddrOffset => SelectedRanges.ToArray().Zip(Stride.ToArray()).Aggregate(0, (acc, t) => acc + t.Item1.Start * t.Item2);

    /// <inheritdoc/>
    public bool Equals(object? other, IEqualityComparer comparer)
    {
        return other is Buffer memRef && EqualityContract == memRef.EqualityContract;
    }

    /// <inheritdoc/>
    public int GetHashCode(IEqualityComparer comparer)
    {
        return EqualityComparer<Type>.Default.GetHashCode(EqualityContract);
    }

    /// <inheritdoc/>
    public override string ToString()
    {
        return $"Buffer({Name}, {ElemType})";
    }

    /// <inheritdoc/>
    protected override bool PrintMembers(StringBuilder builder)
    {
        builder.Append($"Buffer({Name}, {ElemType})");
        return true;
    }
}