// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Reactive;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Affine;

namespace Nncase.Schedule.TileTree;

public sealed class TreeBuilder : ExprVisitor<Unit, Unit>
{
    private readonly Dictionary<Grid, ITreeNode> _memo;
    private readonly int _totalLevel;
    private readonly ScopeNode _rootScopeNode;
    private int _opId;

    private TreeBuilder(int totalLevel)
    {
        _totalLevel = totalLevel;
        _rootScopeNode = new();
        _memo = new();
    }

    public static ScopeNode Build(Grid grid, int totalLevel)
    {
        var builder = new TreeBuilder(totalLevel);
        builder.Visit(grid);
        return builder._rootScopeNode;
    }

    protected override Unit DefaultVisitLeaf(Expr expr) => default;

    protected override Unit VisitLeafGrid(Grid current)
    {
        if (_memo.TryGetValue(current, out var node))
        {
            return default;
        }

        var dependences = new List<OpNode.Dependence>();
        for (int i = 0; i < current.Reads.Length; i++)
        {
            if (current.Reads[i] is Grid producer)
            {
                var producerNode = _memo[producer];
                var producerOp = producerNode.Collect().OfType<OpNode>().First();
                dependences.Add(new OpNode.Dependence(i, producerOp));
            }
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

        var opNode = new OpNode(current, op, copId, dimNames, domain, bufferShapes, dependences.ToArray());
        var tileNodeRoot = new TileNode(_totalLevel, copId, dimNames);
        TileNode tileNodeTail = tileNodeRoot;
        for (int l = _totalLevel - 1; l >= 1; l--)
        {
            var child = new TileNode(l, copId, dimNames);
            tileNodeTail.Child = child;
            tileNodeTail = child;
        }

        tileNodeTail.Child = opNode;
        _rootScopeNode.Add(tileNodeRoot);
        _memo.Add(current, opNode);

        return default;
    }
}
