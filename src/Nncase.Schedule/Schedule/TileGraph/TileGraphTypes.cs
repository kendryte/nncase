// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
using System.Collections;
using System.Collections.Immutable;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using Nncase.Graphs;
using Nncase.IR;
using Nncase.IR.Affine;
using QuikGraph;
using QuikGraph.Collections;

namespace Nncase.Schedule.TileGraph;

public interface ITileable
{
    int Level { get; }

    int OpId { get; }

    /// <summary>
    /// Gets or sets the domain relation which from parent domain map to current node's domain.
    /// </summary>
    DomainRelation DomainRelation { get; set; }
}

public sealed record DomainRelation(int DomainOp, int RangeOp, AffineMap Map)
{
    public DomainRelation ApplyRange(DomainRelation other)
    {
        if (RangeOp != other.DomainOp)
        {
            throw new InvalidOperationException(string.Empty);
        }

        return new DomainRelation(DomainOp, other.RangeOp, Map * other.Map);
    }

    public override string ToString() => $"Op{DomainOp} -> Op{RangeOp}: {Map}";
}

public sealed class TileGrid : ITileable
{
    public TileGrid(Grid grid, Op op, int opId, IEnumerable<string> dimNames, IEnumerable<int> domainBounds, IEnumerable<IEnumerable<int>> bufferShapes)
    {
        Level = 0;
        Grid = grid;
        Op = op;
        OpId = opId;
        DomainRelation = new(opId, opId, AffineMap.Identity(domainBounds.Count()));
        DomainBounds = ImmutableArray.CreateRange(domainBounds);
        BufferShapes = ImmutableArray.CreateRange(bufferShapes.Select(x => ImmutableArray.CreateRange(x)));
    }

    public int Level { get; }

    public int OpId { get; }

    public DomainRelation DomainRelation { get; set; }

    public Grid Grid { get; }

    public Op Op { get; }

    public ImmutableArray<int> DomainBounds { get; }

    public ImmutableArray<ImmutableArray<int>> BufferShapes { get; }

    public ReadOnlySpan<AffineMap> ReadAccesses => Grid.AccessMaps[..^1];

    public AffineMap WriteAccess => Grid.AccessMaps[^1];

    public override string ToString()
    {
        return $"Op{OpId}";
    }
}

[DebuggerDisplay("Op{OpId}@{Level} VertexCount = {VertexCount}, EdgeCount = {EdgeCount}")]
public sealed class TieredTileGraph : TieredAdjacencyGraph<TileGrid, EquatableTaggedEdge<TileGrid, int>>, ITileable
{
    public TieredTileGraph(int topLevel, [NotNull] AdjacencyGraph<TileGrid, EquatableTaggedEdge<TileGrid, int>> wrappedGraph)
        : base(wrappedGraph)
    {
        OpId = -1;
        Level = topLevel;
        DomainRelation = new(-1, -1, IR.Affine.AffineMap.Identity(0));
    }

    public TieredTileGraph([NotNull] TieredTileGraph parentGraph, int level, int opid, DomainRelation relation)
        : base(parentGraph)
    {
        OpId = opid;
        Level = level;
        DomainRelation = relation;
    }

    public int Level { get; }

    public int OpId { get; }

    public DomainRelation DomainRelation { get; set; }

    public override string ToString() => $"Op{OpId}@{Level}";
}
