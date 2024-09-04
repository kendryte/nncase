﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections;
using System.Collections.Immutable;
using System.Reactive;
using Nncase.IR;
using Nncase.IR.Affine;

namespace Nncase.Schedule.TileTree;

public interface ITreeNode
{
    public ITreeNode? Parent { get; set; }

    TReturn Accept<TArg1, TReturn>(ITreeNodeVisitor<TArg1, TReturn> visitor, TArg1 arg1);
}

public interface ITreeNodeVisitor<in TArg1, out TReturn>
{
    TReturn Visit(ScopeNode value, TArg1 arg1);

    TReturn Visit(TileNode value, TArg1 arg1);

    TReturn Visit(OpNode value, TArg1 arg1);
}

public interface IMappableNode
{
    /// <summary>
    /// Gets or sets the domain relation which from parent domain map to current node's domain.
    /// </summary>
    DomainRelation DomainRelation { get; set; }
}

public interface ITileAbleNode : ITreeNode, IMappableNode
{
    int Level { get; }

    int OpId { get; }

    /// <summary>
    /// Gets the domain var names.
    /// </summary>
    ImmutableArray<string> DimNames { get; }
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

public sealed class ScopeNode : ITreeNode
{
    private readonly List<ITreeNode> _children;

    public ScopeNode(ITreeNode? parent = null)
    {
        Parent = parent;
        _children = new();
    }

    public ITreeNode? Parent { get; set; }

    public IList<ITreeNode> Children => _children;

    TReturn ITreeNode.Accept<TArg1, TReturn>(ITreeNodeVisitor<TArg1, TReturn> visitor, TArg1 arg1) => visitor.Visit(this, arg1);

    public void Add(ITreeNode node)
    {
        node.Parent = this;
        _children.Add(node);
    }

    public void Insert(int index, ITreeNode node)
    {
        node.Parent = this;
        _children.Insert(index, node);
    }

    public void InsertRange(int index, IList<ITreeNode> nodes)
    {
        foreach (var item in nodes)
        {
            item.Parent = this;
        }

        _children.InsertRange(index, nodes);
    }

    public void Remove(ITreeNode node)
    {
        _children.Remove(node);
        node.Parent = null;
    }
}

public sealed class TileNode : ITileAbleNode
{
    private ITreeNode _child;

    public TileNode(int level, int opId, IEnumerable<string> dimNames)
    {
        Level = level;
        OpId = opId;
        DimNames = ImmutableArray.CreateRange(dimNames);
        DomainRelation = new(opId, opId, AffineMap.Identity(DimNames.Length));
        _child = null!;
    }

    public ITreeNode? Parent { get; set; }

    public int Level { get; }

    public int OpId { get; }

    /// <summary>
    /// Gets the domain var names.
    /// </summary>
    public ImmutableArray<string> DimNames { get; }

    /// <summary>
    /// Gets or sets the domain relation which from parent domain map to current node's domain.
    /// </summary>
    public DomainRelation DomainRelation { get; set; }

    public ITreeNode Child
    {
        get => _child; set
        {
            _child = value;
            _child.Parent = this;
        }
    }

    TReturn ITreeNode.Accept<TArg1, TReturn>(ITreeNodeVisitor<TArg1, TReturn> visitor, TArg1 arg1) => visitor.Visit(this, arg1);

    public override string ToString()
    {
        return $"Tile{OpId} @ {Level}";
    }
}

public sealed class OpNode : ITileAbleNode
{
    public OpNode(Grid grid, Op op, int opId, IEnumerable<string> dimNames, IEnumerable<int> domainBounds, IEnumerable<IEnumerable<int>> bufferShapes, IEnumerable<Dependence> dependences)
    {
        Level = 0;
        Grid = grid;
        Op = op;
        OpId = opId;
        DimNames = ImmutableArray.CreateRange(dimNames);
        DomainRelation = new(opId, opId, AffineMap.Identity(DimNames.Length));
        DomainBounds = ImmutableArray.CreateRange(domainBounds);
        BufferShapes = ImmutableArray.CreateRange(bufferShapes.Select(x => ImmutableArray.CreateRange(x)));
        Dependences = ImmutableArray.CreateRange(dependences);
    }

    public ITreeNode? Parent { get; set; }

    public int Level { get; }

    public Grid Grid { get; }

    public Op Op { get; }

    public int OpId { get; }

    /// <summary>
    /// Gets the domain var names.
    /// </summary>
    public ImmutableArray<string> DimNames { get; }

    /// <summary>
    /// Gets or sets the domain relation which from parent domain map to current node's domain.
    /// </summary>
    public DomainRelation DomainRelation { get; set; }

    public ImmutableArray<Dependence> Dependences { get; }

    public ImmutableArray<int> DomainBounds { get; }

    public ImmutableArray<ImmutableArray<int>> BufferShapes { get; }

    public ReadOnlySpan<AffineMap> ReadAccesses => Grid.AccessMaps[..^1];

    public AffineMap WriteAccess => Grid.AccessMaps[^1];

    TReturn ITreeNode.Accept<TArg1, TReturn>(ITreeNodeVisitor<TArg1, TReturn> visitor, TArg1 arg1) => visitor.Visit(this, arg1);

    public override string ToString()
    {
        return $"Op{OpId}";
    }

    /// <summary>
    /// this opnode's dependence.
    /// </summary>
    /// <param name="Index"> current read buffer index.</param>
    /// <param name="Node"> producer node. </param>
    public record Dependence(int Index, OpNode Node)
    {
    }
}
