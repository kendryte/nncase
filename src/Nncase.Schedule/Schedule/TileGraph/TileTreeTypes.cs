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
using QuikGraph.Algorithms;
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

    public ImmutableArray<bool> DomainDynamic => _wrapped.DomainDynamic;

    public ImmutableArray<long> DomainBounds => _wrapped.DomainBounds;

    public ImmutableArray<Dimension> DomainBoundExprs => _wrapped.DomainBoundExprs;

    public ImmutableArray<ImmutableArray<long>> BufferShapes => _wrapped.BufferShapes;

    public ReadOnlySpan<AffineMap> ReadAccesses => _wrapped.ReadAccesses;

    public AffineMap WriteAccess => _wrapped.WriteAccess;

    public MicroKernelContext GetMicroKernelContext(ITargetOptions targetOptions) => new(Op, Grid.AccessMaps.ToImmutableArray(), BufferShapes, targetOptions);

    public MicroKernelInfo GetKernelInfo(ITargetOptions targetOptions) => CompilerServices.GetOpMicroKernelInfo(Op, GetMicroKernelContext(targetOptions));

    public TReturn Accept<TArg1, TReturn>(ITreeNodeVisitor<TArg1, TReturn> visitor, TArg1 arg1) => visitor.Visit(this, arg1);

    public override string ToString() => _wrapped.ToString();
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

    public ImmutableArray<bool> DomainDynamic => _wrapped.DomainDynamic;

    public ImmutableArray<Dimension> DomainBoundExprs => _wrapped.DomainBoundExprs;

    public static TileNode FromTileGraph(TieredTileGraph rootGraph, out Dictionary<TieredTileGraph, TileNode> memo)
    {
        memo = new();
        return ConvertToTree(null, rootGraph, rootGraph, memo);
    }

    TReturn ITreeNode.Accept<TArg1, TReturn>(ITreeNodeVisitor<TArg1, TReturn> visitor, TArg1 arg1) => visitor.Visit(this, arg1);

    public override string ToString()
    {
        return _wrapped.ToString();
    }

    private static TileNode ConvertToTree(ITreeNode? parent, TieredTileGraph tileGraph, TieredTileGraph rootGraph, Dictionary<TieredTileGraph, TileNode> memo)
    {
        if (!memo.TryGetValue(tileGraph, out var tnode))
        {
            if (tileGraph.ClustersCount == 0)
            {
                // sort
                var tempGraph = new AdjacencyGraph<TileGrid, Edge<TileGrid>>(allowParallelEdges: false);
                var childVertices = tileGraph.Vertices.ToArray();
                tempGraph.AddVertexRange(childVertices);
                foreach (var edge in rootGraph.Edges)
                {
                    var producers = childVertices.Where(c => c.Equals(edge.Source)).ToArray();
                    var consumers = childVertices.Where(c => c.Equals(edge.Target)).ToArray();
                    foreach (var producer in producers)
                    {
                        foreach (var consumer in consumers)
                        {
                            if (!ReferenceEquals(producer, consumer))
                            {
                                tempGraph.AddEdge(new(producer, consumer));
                            }
                        }
                    }
                }

                tnode = new TileNode(parent, tileGraph, tileGraph.VertexCount);
                int count = 0;
                foreach (var item in tempGraph.TopologicalSort())
                {
                    tnode._children[count++] = new OpNode(tnode, item);
                }
            }
            else
            {
                // sort child clusters
                var tempGraph = new AdjacencyGraph<TieredTileGraph, Edge<TieredTileGraph>>(allowParallelEdges: false);
                var childClusters = tileGraph.Clusters.OfType<TieredTileGraph>().ToArray();
                tempGraph.AddVertexRange(childClusters);
                foreach (var edge in rootGraph.Edges)
                {
                    var producers = childClusters.Where(c => c.ContainsVertex(edge.Source)).ToArray();
                    var consumers = childClusters.Where(c => c.ContainsVertex(edge.Target)).ToArray();
                    foreach (var producer in producers)
                    {
                        foreach (var consumer in consumers)
                        {
                            if (!ReferenceEquals(producer, consumer))
                            {
                                tempGraph.AddEdge(new(producer, consumer));
                            }
                        }
                    }
                }

                tnode = new TileNode(parent, tileGraph, tileGraph.ClustersCount);
                int count = 0;
                foreach (var item in tempGraph.TopologicalSort())
                {
                    tnode._children[count++] = ConvertToTree(tnode, item, rootGraph, memo);
                }
            }

            memo.Add(tileGraph, tnode);
        }

        return tnode;
    }
}
