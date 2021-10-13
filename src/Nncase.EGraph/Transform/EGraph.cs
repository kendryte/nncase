// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using GiGraph.Dot.Entities.Graphs;
using GiGraph.Dot.Entities.Nodes;
using GiGraph.Dot.Extensions;
using GiGraph.Dot.Types.Nodes;
using GiGraph.Dot.Types.Styling;
using GiGraph.Dot.Types.Records;
using GiGraph.Dot.Types.Edges;


namespace Nncase.Transform
{
    /// <summary>
    /// ENode.
    /// </summary>
    public sealed class ENode : IEquatable<ENode?>
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="ENode"/> class.
        /// </summary>
        /// <param name="expr">Expression.</param>
        /// <param name="eclass">EClass.</param>
        /// <param name="children">Children.</param>
        public ENode(Expr expr, EClass eclass, EClass[] children)
        {
            Expr = expr;
            Class = eclass;
            Children = children;
        }

        /// <summary>
        /// Gets or sets expression.
        /// </summary>
        public Expr Expr { get; set; }

        /// <summary>
        /// Gets or sets eclass.
        /// </summary>
        public EClass Class { get; set; }

        /// <summary>
        /// Gets children.
        /// </summary>
        public EClass[] Children { get; }

        /// <summary>
        /// Add current enode information to childrens. 
        /// </summary>
        public void AddUsed()
        {
            foreach (var children in Children)
            {
                children.Used.Add(this);
            }
        }

        public override bool Equals(object? obj)
        {
            return Equals(obj as ENode);
        }

        public bool Equals(ENode? other)
        {
            return other != null &&
                   EqualityComparer<Expr>.Default.Equals(Expr, other.Expr) &&
                   EqualityComparer<EClass[]>.Default.Equals(Children, other.Children);
        }

        public override int GetHashCode()
        {
            return HashCode.Combine(Expr, Children);
        }

        public static bool operator ==(ENode? left, ENode? right)
        {
            return EqualityComparer<ENode>.Default.Equals(left, right);
        }

        public static bool operator !=(ENode? left, ENode? right)
        {
            return !(left == right);
        }
    }

    /// <summary>
    /// EClass.
    /// </summary>
    public sealed class EClass
    {

        public EClass(int id)
        {
            Id = id;
        }
        public int Id { get; }

        private EClass? _parent { get; set; } = null;

        /// <summary>
        /// Gets enodes.
        /// </summary>
        public List<ENode> Nodes { get; } = new List<ENode>();

        public List<ENode> Used { get; } = new List<ENode>();
    }

    /// <summary>
    /// EGraph.
    /// </summary>
    public sealed class EGraph
    {
        private readonly Dictionary<Expr, ENode> _nodes = new Dictionary<Expr, ENode>();
        private readonly List<EClass> _classes = new List<EClass>();

        /// <summary>
        /// Add enode.
        /// </summary>
        /// <param name="expr">Expression.</param>
        /// <returns>Corresponding eclass.</returns>
        public EClass Add(Expr expr)
        {
            if (!_nodes.TryGetValue(expr, out var node))
            {
                var converter = new ENodeConverter(this);
                node = converter.Visit(expr);
            }

            return node.Class;
        }

        private ENode MakeNode(Expr expr, EClass[] children)
        {
            var eclass = new EClass(_classes.Count);
            var node = new ENode(expr, eclass, children);

            node.AddUsed();
            eclass.Nodes.Add(node);

            _nodes.Add(expr, node);
            _classes.Add(eclass);
            return node;
        }



        public DotGraph Dump()
        {
            var g = new DotGraph(directed: true);
            foreach (var eclass in _classes)
            {
                var eclassNode = new DotNode($"{eclass.Id}")
                {
                    Label = $"{eclass.Id}",
                    Shape = DotNodeShape.Circle
                };
                g.Nodes.Add(eclassNode);

                foreach (var enode in eclass.Nodes)
                {
                    string exprId = enode.Expr.GetHashCode().ToString();

                    var args = new List<string> { $"{enode.Expr.GetType().Name}" };
                    for (int i = 0; i < enode.Children.Length; i++)
                    {
                        args.Append($"{exprId}:p{i}");
                    }
                    var exprNodeLabel = new DotRecord(args);
                    var exprNode = g.Nodes.Add(exprId, node =>
                    {
                        node.Label = exprNodeLabel;
                        node.Shape = DotNodeShape.Record;
                    });

                    for (int i = 0; i < enode.Children.Length; i++)
                    {
                        g.Edges.Add(eclassNode, exprNode, edge =>
{
    edge.Head.Endpoint.Port = new DotEndpointPort($"{exprId}:p{i}", DotCompassPoint.NorthEast);
});
                    }

                    //                         recordBuilder.AppendField($"{exprId}:p{i}");
                    //                         // link expr's children with their eclass
                    //                         g.Edges.Add($"{enode.Children[i].Id}", , edge =>
                    //  {
                    //      edge.Head.Endpoint.Port = new DotEndpointPort($"{exprId}:p{i}");
                    //  });

                    // edge eclass with enode
                    g.Edges.Add(exprNode, eclassNode, edge =>
                    {
                        edge.Style.LineStyle = DotLineStyle.Dashed;
                    });
                }
            }
            g.Build();
            return g;
        }

        private sealed class ENodeConverter : ExprVisitor<ENode, IRType>
        {
            private readonly EGraph _graph;

            public ENodeConverter(EGraph graph)
            {
                _graph = graph;
            }

            public override ENode VisitLeaf(Call expr)
            {
                var children = new[] { Visit(expr.Target).Class }.Concat(
                    from p in expr.Parameters select Visit(p).Class).ToArray();
                return _graph.MakeNode(expr, children);
            }

            public override ENode VisitLeaf(Const expr)
            {
                return _graph.MakeNode(expr, Array.Empty<EClass>());
            }

            public override ENode VisitLeaf(Function expr)
            {
                var children = (from p in expr.Parameters select Visit(p).Class)
                    .Concat(new[] { Visit(expr.Body).Class }).ToArray();
                return _graph.MakeNode(expr, children);
            }

            public override ENode VisitLeaf(IR.Tuple expr)
            {
                var children = (from p in expr.Fields select Visit(p).Class).ToArray();
                return _graph.MakeNode(expr, children);
            }

            public override ENode VisitLeaf(Op expr)
            {
                return _graph.MakeNode(expr, Array.Empty<EClass>());
            }

            public override ENode VisitLeaf(Var expr)
            {
                return _graph.MakeNode(expr, Array.Empty<EClass>());
            }
        }
    }
}
