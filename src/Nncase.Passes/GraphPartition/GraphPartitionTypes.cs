// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using Nncase.IR;
using QuikGraph;
using QuikGraph.Graphviz;

namespace Nncase.Passes.GraphPartition;

public enum Compat
{
    UNKNOWN,
    COMPATIBLE,
    INCOMPATIBLE,
}

public enum EdgeTypes
{
    UNKNOWN,
    C2C,
    I2I,
    C2I,
    I2C,
}

public sealed record Vertex
{
    public Vertex(Expr expr, Compat compatType)
    {
        Expr = expr;
        CompatType = compatType;
    }

    public Expr Expr { get; set; }

    public Compat CompatType { get; set; }

    public override string ToString() => Expr.ToString();

    public QuikGraph.Graphviz.Dot.GraphvizColor Color() => CompatType switch
    {
        Compat.INCOMPATIBLE => QuikGraph.Graphviz.Dot.GraphvizColor.Coral,
        Compat.COMPATIBLE => QuikGraph.Graphviz.Dot.GraphvizColor.Olive,
        _ => QuikGraph.Graphviz.Dot.GraphvizColor.Cornsilk,
    };

    public bool Equals(Vertex? other)
    {
        if (other is null)
        {
            return false;
        }

        return ReferenceEquals(Expr, other.Expr) && EqualityComparer<Compat>.Default.Equals(CompatType, other.CompatType);
    }

    public override int GetHashCode()
    {
        return HashCode.Combine(ReferenceEqualityComparer.Instance.GetHashCode(Expr), CompatType.GetHashCode());
    }
}

public sealed record Edge : IEdge<Vertex>
{
    public Edge(EdgeTypes edgeType, Vertex source, Vertex target)
    {
        EdgeType = edgeType;
        Source = source;
        Target = target;
    }

    public EdgeTypes EdgeType { get; set; }

    public Vertex Source { get; set; }

    public Vertex Target { get; set; }

    public bool Equals(Edge? other)
    {
        if (other is null)
        {
            return false;
        }

        return ReferenceEquals(Source, other.Source) &&
        ReferenceEquals(Target, other.Target) &&
        EqualityComparer<EdgeTypes>.Default.Equals(EdgeType, other.EdgeType);
    }

    public override int GetHashCode()
    {
        return HashCode.Combine(ReferenceEqualityComparer.Instance.GetHashCode(Source), ReferenceEqualityComparer.Instance.GetHashCode(Target), EdgeType.GetHashCode());
    }
}

public sealed class Graph : AdjacencyGraph<Vertex, Edge>
{
    public void DumpDot(string fullPathName)
    {
        using (var writer = new StreamWriter(fullPathName))
        {
            var a = this.ToGraphviz<Vertex, Edge>(algorithm =>
            {
                algorithm.FormatVertex += (_, args) => args.VertexFormat.Label = args.Vertex.ToString();
                algorithm.FormatVertex += (_, args) => args.VertexFormat.Style = QuikGraph.Graphviz.Dot.GraphvizVertexStyle.Filled;
                algorithm.FormatVertex += (_, args) => args.VertexFormat.FillColor = args.Vertex.Color();
            });
            writer.Write(a);
        }
    }
}

public sealed record Subgraph(int Index, List<Vertex> Nodes, List<Edge> InputEdges, List<Edge> OutputEdges, List<Edge> InteriorEdges);
