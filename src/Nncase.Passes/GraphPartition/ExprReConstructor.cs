// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reactive;
using Nncase.Graphs;
using Nncase.IR;
using Nncase.IR.Affine;
using QuikGraph;

namespace Nncase.Passes.GraphPartition;

public class ExprReconstructor<TVertex, TEdge>
    where TVertex : IExprVertex
    where TEdge : class, IExprEdge<TVertex>
{
    public ExprReconstructor(CondensationGraphAlgorithm<TVertex, TEdge> algo)
    {
        Algo = algo;
        ClusterMemo = new(ReferenceEqualityComparer.Instance);
    }

    public CondensationGraphAlgorithm<TVertex, TEdge> Algo { get; }

    protected Dictionary<ClusteredBidirectionalGraph<TVertex, TEdge>, BaseExpr> ClusterMemo { get; }

    public BaseExpr Construct()
    {
        var dfsVisitor = new QuikGraph.Algorithms.TopologicalSort.SourceFirstTopologicalSortAlgorithm<ClusteredBidirectionalGraph<TVertex, TEdge>, Edge<ClusteredBidirectionalGraph<TVertex, TEdge>>>(Algo.CondensedGraph);
        dfsVisitor.Compute();
        for (var i = 0; i < dfsVisitor.SortedVertices.Length; i++)
        {
            ClusterMemo.Add(dfsVisitor.SortedVertices[i], OnFinishCluster(dfsVisitor.SortedVertices[i], i));
        }

        return ClusterMemo[dfsVisitor.SortedVertices[^1]];
    }

    protected virtual IEnumerable<(BaseExpr Pre, BaseExpr Post)> GetClusterArgumentPairs(ClusteredBidirectionalGraph<TVertex, TEdge> cluster)
    {
        var pairs = new List<(BaseExpr Pre, BaseExpr Post)>();
        foreach (var inEdge in cluster.InEdges(Algo.ClusteredGraph))
        {
            // get in Expr
            BaseExpr postArg;
            var sourceCluster = Algo.VertexMap[inEdge.Source];
            var sourceOutVertices = sourceCluster.OutVertices(Algo.ClusteredGraph).ToArray();
            if (sourceOutVertices.Length == 1)
            {
                postArg = ClusterMemo[sourceCluster];
            }
            else
            {
                var sourceOutIndex = sourceOutVertices.IndexOf(inEdge.Source);
                var postResult = ClusterMemo[sourceCluster];
                postArg = postResult[sourceOutIndex];
            }

            pairs.Add((inEdge.Source.Expr, postArg));
        }

        return pairs;
    }

    protected virtual BaseExpr OnFinishCluster(ClusteredBidirectionalGraph<TVertex, TEdge> cluster, int sortIndex)
    {
        return cluster.VertexCount == 1 ? OnAtomCluster(cluster, sortIndex) : OnComplexCluster(cluster, sortIndex);
    }

    protected virtual BaseExpr OnAtomCluster(ClusteredBidirectionalGraph<TVertex, TEdge> cluster, int sortIndex)
    {
        var pairs = GetClusterArgumentPairs(cluster);
        var cloner = new ExprClusterCloner(pairs.ToDictionary(p => p.Pre, p => p.Post, new ReferenceEqualityComparer<BaseExpr>()));
        return cloner.Clone(cluster.Vertices.First().Expr, default);
    }

    protected virtual BaseExpr OnComplexCluster(ClusteredBidirectionalGraph<TVertex, TEdge> cluster, int sortIndex)
    {
        var pairs = GetClusterArgumentPairs(cluster);
        var cloner = new ExprClusterCloner(pairs.ToDictionary(p => p.Pre, p => p.Post, new ReferenceEqualityComparer<BaseExpr>()));
        var outVertices = cluster.OutVertices().ToArray();
        if (outVertices.Length == 1)
        {
            return cloner.Clone(outVertices[0].Expr, default);
        }
        else
        {
            var fields = new List<BaseExpr>();
            foreach (var outVertex in outVertices)
            {
                fields.Add(cloner.Clone(outVertex.Expr, default));
            }

            return new IR.Tuple(fields.ToArray());
        }
    }
}

public class ExprClusterCloner : ExprCloner<Unit>
{
    public ExprClusterCloner(Dictionary<BaseExpr, BaseExpr> extractMemo)
    {
        ExtractMemo = extractMemo;
    }

    public Dictionary<BaseExpr, BaseExpr> ExtractMemo { get; }

    protected override BaseExpr DispatchVisit(BaseExpr expr, Unit context)
    {
        if (HasVisited(expr, out var result))
        {
            return result;
        }

        if (ExtractMemo.TryGetValue(expr, out var @param))
        {
            return MarkVisited(expr, @param);
        }

        return MarkVisited(expr, base.DispatchVisit(expr, context));
    }

    protected override Expr VisitLeafFunction(Function expr, Unit context) => expr;

    protected override Expr VisitLeafVar(Var expr, Unit context) => expr;

    protected override BaseExpr VisitLeafDimVar(DimVar expr, Unit context) => expr;

    protected override Expr VisitLeafConst(Const expr, Unit context) => expr;

    protected override Expr VisitLeafNone(None expr, Unit context) => expr;

    protected override Expr VisitLeafOp(Op expr, Unit context) => expr;
}
