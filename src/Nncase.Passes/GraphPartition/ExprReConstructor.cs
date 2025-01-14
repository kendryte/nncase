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

public class ExprReConstructor<TVertex, TEdge>
    where TVertex : IExprVertex
    where TEdge : class, IExprEdge<TVertex>
{
    public ExprReConstructor(CondensationGraphAlgorithm<TVertex, TEdge> algo)
    {
        Algo = algo;
        ClusterMemo = new(ReferenceEqualityComparer.Instance);
    }

    public CondensationGraphAlgorithm<TVertex, TEdge> Algo { get; }

    protected Dictionary<ClusteredBidirectionalGraph<TVertex, TEdge>, Expr> ClusterMemo { get; }

    public Expr Construct()
    {
        var dfsVisitor = new QuikGraph.Algorithms.TopologicalSort.SourceFirstTopologicalSortAlgorithm<ClusteredBidirectionalGraph<TVertex, TEdge>, Edge<ClusteredBidirectionalGraph<TVertex, TEdge>>>(Algo.CondensedGraph);
        dfsVisitor.Compute();
        for (var i = 0; i < dfsVisitor.SortedVertices.Length; i++)
        {
            ClusterMemo.Add(dfsVisitor.SortedVertices[i], OnFinishCluster(dfsVisitor.SortedVertices[i], i));
        }

        return ClusterMemo[dfsVisitor.SortedVertices[^1]];
    }

    protected IEnumerable<(Expr Pre, Expr Post)> GetClusterArgumentPairs(ClusteredBidirectionalGraph<TVertex, TEdge> cluster)
    {
        var pairs = new List<(Expr Pre, Expr Post)>();
        foreach (var inEdge in cluster.InEdges(Algo.ClusteredGraph))
        {
            // get in Expr
            Expr postArg;
            var sourceCluster = Algo.VertexMap[inEdge.Source];
            var sourcerOutVertices = sourceCluster.OutVertices().ToArray();
            if (sourcerOutVertices.Length == 1)
            {
                postArg = ClusterMemo[sourceCluster];
            }
            else
            {
                var sourceOutIndex = sourcerOutVertices.IndexOf(inEdge.Source);
                postArg = IR.F.Tensors.GetItem(ClusterMemo[sourceCluster], sourceOutIndex);
            }

            pairs.Add((inEdge.Source.Expr, postArg));
        }

        return pairs;
    }

    protected virtual Expr OnFinishCluster(ClusteredBidirectionalGraph<TVertex, TEdge> cluster, int sortIndex)
    {
        return cluster.VertexCount == 1 ? OnAtomCluster(cluster, sortIndex) : OnComplexCluster(cluster, sortIndex);
    }

    protected virtual Expr OnAtomCluster(ClusteredBidirectionalGraph<TVertex, TEdge> cluster, int sortIndex)
    {
        var pairs = GetClusterArgumentPairs(cluster);
        var cloner = new ExprClusterCloner(pairs.ToDictionary(p => p.Pre, p => p.Post, new ReferenceEqualityComparer<Expr>()));
        return cloner.Clone(cluster.Vertices.First().Expr, default);
    }

    protected virtual Expr OnComplexCluster(ClusteredBidirectionalGraph<TVertex, TEdge> cluster, int sortIndex)
    {
        var pairs = GetClusterArgumentPairs(cluster);
        var cloner = new ExprClusterCloner(pairs.ToDictionary(p => p.Pre, p => p.Post, new ReferenceEqualityComparer<Expr>()));
        var outVertices = cluster.OutVertices().ToArray();
        if (outVertices.Length == 1)
        {
            return cloner.Clone(outVertices[0].Expr, default);
        }
        else
        {
            var fields = new List<Expr>();
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
    public ExprClusterCloner(Dictionary<Expr, Expr> extractMemo)
    {
        ExtractMemo = extractMemo;
    }

    public Dictionary<Expr, Expr> ExtractMemo { get; }

    protected override Expr DispatchVisit(Expr expr, Unit context)
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

    protected override Expr VisitLeafVar(Var expr, Unit context)
    {
        return expr;
    }

    protected override Expr VisitLeafConst(Const expr, Unit context)
    {
        return expr;
    }

    protected override Expr VisitLeafNone(None expr, Unit context)
    {
        return expr;
    }

    protected override Expr VisitLeafOp(Op expr, Unit context)
    {
        return expr;
    }
}
