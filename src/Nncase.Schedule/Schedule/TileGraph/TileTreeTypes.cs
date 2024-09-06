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

public interface ITreeNode : ITileable
{
    ITreeNode? Parent { get; }

    TReturn Accept<TArg1, TReturn>(ITreeNodeVisitor<TArg1, TReturn> visitor, TArg1 arg1);
}

public interface ITreeNodeVisitor<in TArg1, out TReturn>
{
    TReturn Visit(TileNode value, TArg1 arg1);

    TReturn Visit(OpNode value, TArg1 arg1);
}

public sealed class OpNode : ITreeNode
{
    private readonly TileGrid _wrapped;

    public OpNode(ITreeNode? parent, TileGrid wrapped)
    {
        Parent = parent;
        _wrapped = wrapped;
    }

    public TileGrid Wrapped => _wrapped;

    public int Level => _wrapped.Level;

    public int OpId => _wrapped.OpId;

    public DomainRelation DomainRelation { get => _wrapped.DomainRelation; set => throw new NotSupportedException(); }

    public ITreeNode? Parent { get; }

    public Grid Grid => _wrapped.Grid;

    public Op Op => _wrapped.Op;

    public ImmutableArray<int> DomainBounds => _wrapped.DomainBounds;

    public ImmutableArray<ImmutableArray<int>> BufferShapes => _wrapped.BufferShapes;

    public ReadOnlySpan<AffineMap> ReadAccesses => _wrapped.ReadAccesses;

    public AffineMap WriteAccess => _wrapped.WriteAccess;

    public MicroKernelInfo GetKernelInfo(ITargetOptions targetOptions) => CompilerServices.GetOpMicroKernelInfo(Op, new(Grid.AccessMaps.ToImmutableArray(), BufferShapes, targetOptions));

    public TReturn Accept<TArg1, TReturn>(ITreeNodeVisitor<TArg1, TReturn> visitor, TArg1 arg1) => visitor.Visit(this, arg1);
}

public sealed class TileNode : ITreeNode
{
    private readonly TieredTileGraph _wrapped;
    private readonly ITreeNode[] _children;

    public TileNode(ITreeNode? parent, TieredTileGraph wrapped, ITreeNode[] children)
    {
        Parent = parent;
        _wrapped = wrapped;
        _children = children.ToArray();
    }

    private TileNode(ITreeNode? parent, TieredTileGraph wrapped, int childCount)
    {
        Parent = parent;
        _wrapped = wrapped;
        _children = new ITreeNode[childCount];
    }

    public ITreeNode? Parent { get; private set; }

    public TieredTileGraph Wrapped => _wrapped;

    public ReadOnlySpan<ITreeNode> Children => _children;

    public int Level => _wrapped.Level;

    public int OpId => _wrapped.OpId;

    public DomainRelation DomainRelation { get => _wrapped.DomainRelation; set => throw new NotSupportedException(); }

    public static TileNode FromTileGraph(TieredTileGraph rootGraph, out Dictionary<TieredTileGraph, TileNode> memo)
    {
        memo = new();
        return ConvertToTree(null, rootGraph, memo);
    }

    TReturn ITreeNode.Accept<TArg1, TReturn>(ITreeNodeVisitor<TArg1, TReturn> visitor, TArg1 arg1) => visitor.Visit(this, arg1);

    public override string ToString()
    {
        return _wrapped.ToString();
    }

    private static TileNode ConvertToTree(ITreeNode? parent, TieredTileGraph tileGraph, Dictionary<TieredTileGraph, TileNode> memo)
    {
        if (!memo.TryGetValue(tileGraph, out var tnode))
        {
            if (tileGraph.ClustersCount == 0)
            {
                tnode = new TileNode(parent, tileGraph, tileGraph.VertexCount);
                int count = 0;
                foreach (var item in tileGraph.Vertices)
                {
                    tnode._children[count++] = new OpNode(tnode, item);
                }
            }
            else
            {
                tnode = new TileNode(parent, tileGraph, tileGraph.ClustersCount);
                int count = 0;
                foreach (var item in tileGraph.Clusters.OfType<TieredTileGraph>())
                {
                    tnode._children[count++] = ConvertToTree(tnode, item, memo);
                }
            }

            memo.Add(tileGraph, tnode);
        }

        return tnode;
    }
}
