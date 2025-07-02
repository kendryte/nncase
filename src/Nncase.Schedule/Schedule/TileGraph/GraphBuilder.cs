// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Linq;
using System.Reactive;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.Utilities;
using QuikGraph;
using Isl = IntegerSetLibrary;

namespace Nncase.Schedule.TileGraph;

public sealed class GraphBuilder : ExprVisitor<Unit, Unit>
{
    private readonly Dictionary<Grid, TileGrid> _memo;
    private readonly Dictionary<Grid, TieredTileGraph> _exprMemo;
    private readonly int _totalLevel;
    private int _opId;

    public GraphBuilder(int topLevel)
    {
        _totalLevel = topLevel;
        RootGraph = new(-1, new AdjacencyGraph<TileGrid, EquatableTaggedEdge<TileGrid, int>>());
        _memo = new();
        _exprMemo = new();
    }

    public TieredTileGraph RootGraph { get; }

    public static TieredTileGraph Build(BaseExpr expr, int topLevel, out Dictionary<Grid, TieredTileGraph> exprMemo)
    {
        var builder = new GraphBuilder(topLevel);
        builder.Visit(expr);
        exprMemo = builder._exprMemo;
        return builder.RootGraph;
    }

    protected override Unit DefaultVisitLeaf(BaseExpr expr) => default;

    protected override Unit VisitLeafGrid(Grid current)
    {
        if (_memo.TryGetValue(current, out var node))
        {
            return default;
        }

        var bufferShapeValues = current.Buffers.AsValueEnumerable().Select(b => TilingUtilities.GetBufferShape(b, true).ToValueArray()).ToArray();
        var bufferShapes = current.Buffers.AsValueEnumerable().Select(b => TilingUtilities.GetBufferShape(b, false)).ToArray();
        var bufferExprs = Enumerable.Range(0, current.Buffers.Length).Select(current.GetArgument).ToArray();
        Isl.set[] bufferDomains;
        HashSet<DimVar> dimVars = new();
        {
            var tps = bufferShapes.AsValueEnumerable().Select(shape => (ISLUtility.ToDomain(shape, out var paramMap), paramMap)).ToArray();
            bufferDomains = tps.Select(t => t.Item1).ToArray();
            dimVars.UnionWith(tps.Select(t => t.paramMap).SelectMany(i => i).ToArray());
        }

        var accessMaps = current.AccessMaps.AsValueEnumerable().Select(AffineUtility.AsMap).ToArray();
        var (domain, domainDynamic, domainBoundValues, domainBoundExprs) = TilingUtilities.InferDomainBounds(bufferExprs, bufferDomains, accessMaps, dimVars);

        var copId = _opId++;
        var domainDims = current.AccessMaps[0].Domains.Length;
        var dimNames = Enumerable.Range(0, domainDims).Select(i => $"Op{copId}_d{i}").ToArray();
        if (current.Body[0] is not Call { Target: Op op })
        {
            throw new InvalidOperationException("body is not call");
        }

        var opNode = new TileGrid(current, op, copId, dimNames, domainBoundValues, domainBoundExprs, domainDynamic, bufferShapeValues);

        var tileNodeRoot = RootGraph.CreateCluster<TieredTileGraph>(_totalLevel, copId, new DomainRelation(copId, copId, AffineMap.Identity(domainDims)), domainBoundExprs, domainDynamic);
        var tileNodeTail = tileNodeRoot;
        for (int l = _totalLevel - 1; l >= 1; l--)
        {
            tileNodeTail = tileNodeTail.CreateCluster<TieredTileGraph>(l, copId, new DomainRelation(copId, copId, AffineMap.Identity(domainDims)), domainBoundExprs, domainDynamic);
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
