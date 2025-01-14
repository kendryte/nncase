// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reactive;
using Nncase.IR;
using Nncase.IR.Affine;
using QuikGraph;

namespace Nncase.Passes.GraphPartition;

public interface IExprVertex
{
    Expr Expr { get; }

    static abstract IExprVertex Create(Expr expr);
}

public interface IExprEdge<TExprVertex> : IEdge<TExprVertex>
    where TExprVertex : IExprVertex
{
    static abstract IExprEdge<TExprVertex> Create(TExprVertex source, TExprVertex target, int index);
}

public sealed record ExprVertex : IExprVertex
{
    private ExprVertex(Expr expr)
    {
        Expr = expr;
    }

    public Expr Expr { get; }

    public static IExprVertex Create(Expr expr) => new ExprVertex(expr);

    public bool Equals(ExprVertex? other)
    {
        return ReferenceEquals(this, other);
    }

    public override int GetHashCode() => ReferenceEqualityComparer.Instance.GetHashCode(this);
}

public sealed record ExprEdge : IExprEdge<ExprVertex>
{
    private ExprEdge(ExprVertex source, ExprVertex target, int index)
    {
        Source = source;
        Target = target;
        Index = index;
    }

    public ExprVertex Source { get; }

    public ExprVertex Target { get; }

    public int Index { get; }

    public static IExprEdge<ExprVertex> Create(ExprVertex source, ExprVertex target, int index) => new ExprEdge(source, target, index);
}

public class ExprGraphConvertor<TVertex, TEdge> : ExprVisitor<TVertex, Unit, IMutableVertexAndEdgeListGraph<TVertex, TEdge>>
    where TVertex : IExprVertex
    where TEdge : IExprEdge<TVertex>
{
    protected override TVertex DefaultVisitLeaf(Expr expr, IMutableVertexAndEdgeListGraph<TVertex, TEdge> graph)
    {
        var target = (TVertex)TVertex.Create(expr);
        graph.AddVertex(target);
        int count = 0;
        foreach (var item in expr.Operands)
        {
            var source = Visit(item, graph);
            var edge = (TEdge)TEdge.Create(source, target, count++);
            graph.AddEdge(edge);
        }

        return target;
    }
}
