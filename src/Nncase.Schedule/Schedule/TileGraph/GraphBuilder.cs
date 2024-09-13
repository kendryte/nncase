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
    private readonly Dictionary<Grid, TileGrid> _memo;
    private readonly Dictionary<Grid, TieredTileGraph> _exprMemo;
    private readonly int _totalLevel;
    private int _opId;

    public GraphBuilder(int totalLevel)
    {
        _totalLevel = totalLevel;
        RootGraph = new(totalLevel + 1, new AdjacencyGraph<TileGrid, EquatableTaggedEdge<TileGrid, int>>());
        _memo = new();
        _exprMemo = new();
    }

    public TieredTileGraph RootGraph { get; }

    public static TieredTileGraph Build(Expr expr, int totalLevel, out Dictionary<Grid, TieredTileGraph> exprMemo)
    {
        var builder = new GraphBuilder(totalLevel);
        builder.Visit(expr);
        exprMemo = builder._exprMemo;
        return builder.RootGraph;
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

        var opNode = new TileGrid(current, op, copId, dimNames, domain, bufferShapes);

        var tileNodeRoot = RootGraph.CreateCluster<TieredTileGraph>(_totalLevel, copId, new DomainRelation(copId, copId, AffineMap.Identity(domainDims)));
        var tileNodeTail = tileNodeRoot;
        for (int l = _totalLevel - 1; l >= 1; l--)
        {
            tileNodeTail = tileNodeTail.CreateCluster<TieredTileGraph>(l, copId, new DomainRelation(copId, copId, AffineMap.Identity(domainDims)));
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
        _exprMemo.Add(current, tileNodeRoot);

        return default;
    }
}
