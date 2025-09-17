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

public sealed class TieredTileGraphBuilder : ExprVisitor<Unit, Unit>
{
    private readonly Dictionary<Grid, TileGrid> _memo;
    private readonly Dictionary<Grid, TieredTileGraph> _exprMemo;
    private int _opId;

    public TieredTileGraphBuilder(int levelCount)
    {
        RootGraph = new(new AdjacencyGraph<TileGrid, EquatableTaggedEdge<TileGrid, int>>());
        _memo = new();
        _exprMemo = new();
        LevelCount = levelCount;
    }

    public TieredTileGraph RootGraph { get; }

    public int LevelCount { get; }

    public static TieredTileGraph Build(BaseExpr expr, int levelCount, out Dictionary<Grid, TieredTileGraph> exprMemo)
    {
        var builder = new TieredTileGraphBuilder(levelCount);
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

        /*
            note currently we're not use the affine map's extents for build domain.
            so the domain we built is not consider the primtive shape. wo also can't use extent when building ast.
        */
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

        var opNode = new TileGrid(current, op, copId, domainBoundValues, new DomainRelation(copId, copId, AffineMap.Identity(domainDims)), domainBoundExprs, domainDynamic, bufferShapeValues);

        var tileNodeRoot = RootGraph.CreateCluster<TieredTileGraph>(LevelCount - 1, copId, new DomainRelation(copId, copId, AffineMap.Identity(domainDims)), domainBoundExprs, domainDynamic);
        var tileNodeTail = tileNodeRoot;
        for (int l = LevelCount - 2; l >= 0; l--)
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
