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

        var bufferShapes = current.Buffers.AsValueEnumerable().Select(TilingUtilities.GetBufferShape).ToArray();
        var bufferDomains = current.Buffers.AsValueEnumerable().Select(b => ISLUtility.AsDomain(b.CheckedShape)).ToArray();
        var accessMaps = current.AccessMaps.AsValueEnumerable().Select(AffineUtility.AsIslMap).ToArray();
        var domain = TilingUtilities.InferDomainBounds(bufferDomains, accessMaps);

        var domainBounds = new long[domain.n_dim()];
        var domainDynamic = new bool[domain.n_dim()];
        for (int i = 0; i < domain.n_dim(); i++)
        {
            var maxAff = domain.dim_max(i);
            domainBounds[i] = domain.dim_max_val(i).num_si() + 1; // +1 for exclusive upper bound
            if (!maxAff.is_cst())
            {
                domainDynamic[i] = true;
            }
        }

        // get the domain bounds expression.
        var reversedAccessMaps = accessMaps.Select(m => m.intersect_domain(domain).reverse()).ToArray();
        var shapeToDomainMap = reversedAccessMaps.Skip(1).Aggregate(reversedAccessMaps.Take(1).Single(), (acc, value) => acc.flat_domain_product(value));
        var shapeIdMap = bufferShapes.Select((shape, i) => shape.Select((s, j) => ($"d{i}_{j}", (Dimension)new IR.DimAt(new IR.Shapes.ShapeOf(current.GetArgument(i)), j)
        {
            Metadata = new() { Range = new(0, domainBounds[i]) },
        }))).SelectMany(s => s).ToDictionary(p => p.Item1, p => p.Item2);
        var astBuild = Isl.ast_build.from_context(new Isl.set(Isl.ctx.Current, $"{{ [{string.Join(',', shapeIdMap.Keys)}]:  }}"));
        var shapeToDomainAstExpr = astBuild.access_from(shapeToDomainMap.lexmax_pw_multi_aff());

        var domainBoundsExpr = new Dimension[domainBounds.Length];
        for (int i = 0; i < domainBounds.Length; i++)
        {
            if (domainDynamic[i])
            {
                var dimExpr = ISLUtility.AsDimension(shapeToDomainAstExpr.op_arg(1 + i), shapeIdMap);
                var dimUpperBoundAff = domain.max_multi_pw_aff().at(i);

                // the upper bound's bounds.
                dimExpr.Metadata = new()
                {
                    Range = new(dimUpperBoundAff.min_val().num_si() + 1, dimUpperBoundAff.max_val().num_si() + 1),
                };
                domainBoundsExpr[i] = dimExpr;
            }
            else
            {
                domainBoundsExpr[i] = domainBounds[i];
            }
        }

        var copId = _opId++;
        var domainDims = current.AccessMaps[0].Domains.Length;
        var dimNames = Enumerable.Range(0, domainDims).Select(i => $"Op{copId}_d{i}").ToArray();
        if (current.Body[0] is not Call { Target: Op op })
        {
            throw new InvalidOperationException("body is not call");
        }

        var opNode = new TileGrid(current, op, copId, dimNames, domainBounds, domainBoundsExpr, domainDynamic, bufferShapes);

        var tileNodeRoot = RootGraph.CreateCluster<TieredTileGraph>(_totalLevel, copId, new DomainRelation(copId, copId, AffineMap.Identity(domainDims)), domainBoundsExpr, domainDynamic);
        var tileNodeTail = tileNodeRoot;
        for (int l = _totalLevel - 1; l >= 1; l--)
        {
            tileNodeTail = tileNodeTail.CreateCluster<TieredTileGraph>(l, copId, new DomainRelation(copId, copId, AffineMap.Identity(domainDims)), domainBoundsExpr, domainDynamic);
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
