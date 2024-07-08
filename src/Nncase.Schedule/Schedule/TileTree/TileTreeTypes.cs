// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections;
using System.Reactive;
using Nncase.IR;
using Nncase.IR.Affine;
using VisitorPatternGenerator;

namespace Nncase.Schedule.TileTree;

public partial interface ITreeNode
{
    public ITreeNode? Parent { get; set; }
}

[Visitor<ITreeNode>]
public partial interface ITreeNodeVisitor<in TArg1, out TReturn>
{
}

public interface ITileAbleNode : ITreeNode
{
    int Level { get; }

    int OpId { get; }

    /// <summary>
    /// Gets the domain var names.
    /// </summary>
    string[] DimNames { get; }

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
}

[Acceptor<ITreeNode, ScopeNode>]
public sealed partial class ScopeNode
{
    private readonly List<ITreeNode> _children;

    public ScopeNode(ITreeNode? parent = null)
    {
        Parent = parent;
        _children = new();
    }

    public ITreeNode? Parent { get; set; }

    public IList<ITreeNode> Children => _children;

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

[Acceptor<ITreeNode, TileNode>]
public sealed partial class TileNode : ITileAbleNode
{
    private ITreeNode _child;

    public TileNode(int level, int opId, string[] vars)
    {
        Level = level;
        OpId = opId;
        DimNames = vars;
        DomainRelation = new(opId, opId, AffineMap.Identity(vars.Length));
        _child = null!;
    }

    public ITreeNode? Parent { get; set; }

    public int Level { get; }

    public int OpId { get; }

    /// <summary>
    /// Gets the domain var names.
    /// </summary>
    public string[] DimNames { get; }

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

    public override string ToString()
    {
        return $"Tile{OpId} @ {Level}";
    }
}

[Acceptor<ITreeNode, OpNode>]
public sealed partial class OpNode : ITileAbleNode
{
    private readonly Expr[] _buffers;
    private readonly AffineMap[] _accessMaps;

    public OpNode(Grid grid, Op op, Expr[] buffers, int opId, string[] domainNames, int[] domain, int[][] bufferShapes, AffineMap[] accessMaps, Dependence[] dependences)
    {
        Level = 0;
        Grid = grid;
        Op = op;
        _buffers = buffers;
        OpId = opId;
        DimNames = domainNames;
        DomainRelation = new(opId, opId, AffineMap.Identity(domainNames.Length));
        DomainBounds = domain;
        BufferShapes = bufferShapes;
        Dependences = dependences;
        _accessMaps = accessMaps;
    }

    public ITreeNode? Parent { get; set; }

    public int Level { get; }

    public Grid Grid { get; }

    public Op Op { get; }

    public ReadOnlySpan<Expr> Buffers => _buffers;

    public int OpId { get; }

    /// <summary>
    /// Gets the domain var names.
    /// </summary>
    public string[] DimNames { get; }

    /// <summary>
    /// Gets or sets the domain relation which from parent domain map to current node's domain.
    /// </summary>
    public DomainRelation DomainRelation { get; set; }

    public IReadOnlyList<Dependence> Dependences { get; }

    public IReadOnlyList<int> DomainBounds { get; }

    public int[][] BufferShapes { get; }

    public ReadOnlySpan<AffineMap> ReadAccesses => _accessMaps.AsSpan()[..^1];

    public AffineMap WriteAccess => _accessMaps[^1];

    public ReadOnlySpan<AffineMap> AccessMaps => _accessMaps;

    public override string ToString()
    {
        return $"Op{OpId}";
    }

    /// <summary>
    /// index is current read buffer index.
    /// </summary>
    public record Dependence(int Index, OpNode Node)
    {
    }
}
