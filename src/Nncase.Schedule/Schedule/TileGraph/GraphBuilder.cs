// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Reactive;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Affine;
using QuikGraph;

namespace Nncase.Schedule.TileGraph;

public sealed class GraphBuilder : ExprVisitor<Unit, Unit>
{
    private readonly Dictionary<Grid, OpNode> _memo;
    private readonly int _totalLevel;
    private int _opId;

    public GraphBuilder(int totalLevel)
    {
        _totalLevel = totalLevel;
        OpGraph = new AdjacencyGraph<OpNode, OpEdge>();
        RootGraph = new(totalLevel + 1, OpGraph);
        _memo = new();
    }

    public AdjacencyGraph<OpNode, OpEdge> OpGraph { get; }

    public TileGraph RootGraph { get; }

    public static void Build(Grid grid, int totalLevel)
    {
        var builder = new GraphBuilder(totalLevel);
        builder.Visit(grid);
    }

    protected override Unit DefaultVisitLeaf(Expr expr) => default;

    protected override Unit VisitLeafGrid(Grid current)
    {
        if (_memo.TryGetValue(current, out var node))
        {
            return default;
        }

        var bufferShapes = current.Buffers.AsValueEnumerable().Select(TilingUtilities.GetBufferShape).ToArray();
        var domain = TilingUtilities.InferDomainBounds(bufferShapes, current.AccessMaps.ToArray());
        var copId = _opId++;
        var domainDims = current.AccessMaps[0].Domains.Length;
        var dimNames = Enumerable.Range(0, domainDims).Select(i => $"Op{copId}_d{i}").ToArray();
        if (current.Body[0] is not Call { Target: Op op })
        {
            throw new InvalidOperationException("body is not call");
        }

        var opNode = new OpNode(current, op, copId, dimNames, domain, bufferShapes);

        var tileNodeRoot = RootGraph.CreateCluster<TileGraph>(_totalLevel, copId, new DomainRelation(copId, copId, AffineMap.Identity(domainDims)));
        var tileNodeTail = tileNodeRoot;
        for (int l = _totalLevel - 1; l >= 1; l--)
        {
            tileNodeRoot.AddVertex(opNode);
            tileNodeTail = tileNodeRoot.CreateCluster<TileGraph>(l, copId, new DomainRelation(copId, copId, AffineMap.Identity(domainDims)));
        }

        tileNodeTail.AddVertex(opNode);

        for (int i = 0; i < current.Reads.Length; i++)
        {
            if (current.Reads[i] is Grid producer)
            {
                var producerNode = _memo[producer];
                RootGraph.AddEdge(new(producerNode, opNode, i));
            }
        }

        _memo.Add(current, opNode);

        return default;
    }
}
