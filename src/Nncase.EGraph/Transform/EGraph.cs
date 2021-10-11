// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.Transform
{
    /// <summary>
    /// ENode.
    /// </summary>
    public sealed class ENode
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
    }

    /// <summary>
    /// EClass.
    /// </summary>
    public sealed class EClass
    {
        /// <summary>
        /// Gets enodes.
        /// </summary>
        public List<ENode> Nodes { get; } = new List<ENode>();
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
            var eclass = new EClass();
            _classes.Add(eclass);
            var node = new ENode(expr, eclass, children);
            eclass.Nodes.Add(node);
            _nodes.Add(expr, node);
            return node;
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
