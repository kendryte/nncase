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
    public sealed class ENode : IEquatable<ENode?>
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="ENode"/> class.
        /// </summary>
        /// <param name="expr">Expression.</param>
        /// <param name="children">Children.</param>
        public ENode(Expr expr, EClass[] children)
        {
            Expr = expr;
            Children = children;
        }

        /// <summary>
        /// Gets or sets expression.
        /// </summary>
        public Expr Expr { get; set; }

        /// <summary>
        /// Gets children.
        /// </summary>
        public EClass[] Children { get; }

        /// <summary>
        /// Add current enode information to childrens. 
        /// </summary>
        public void AddUsed(EClass eClass)
        {
            foreach (var children in Children)
            {
                children.Used.Add((this, eClass));
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

        public ENode Canonicalize()
        {
            return new ENode(Expr, (from c in Children select c.Find()).ToArray());
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
        public List<ENode> Nodes { get; set; } = new List<ENode>();

        public List<(ENode, EClass)> Used { get; set; } = new List<(ENode, EClass)>();

        public EClass Find()
        {
            if (_parent is null)
            {
                return this;
            }
            _parent = Find();
            return _parent;
        }
    }

    /// <summary>
    /// EGraph.
    /// </summary>
    public sealed class EGraph
    {
        private readonly Dictionary<ENode, EClass> _hascons = new Dictionary<ENode, EClass>();

        private readonly List<EClass> _classes = new List<EClass>();
        private int _version = 0;

        private readonly List<EClass> _worklist = new List<EClass>();

        public List<EClass> Classes => _classes;
        public Dictionary<ENode, EClass> Nodes => _hascons;

        public int Version => _version;

        /// <summary>
        /// Add enode.
        /// </summary>
        /// <param name="expr">Expression.</param>
        /// <returns>Corresponding eclass.</returns>
        public EClass Add(Expr expr)
        {
            var converter = new ENodeConverter(this);
            return converter.Visit(expr);
        }

        private EClass MakeNode(Expr expr, EClass[] children)
        {
            // todo rewrite it.
            // EClass eclass = new EClass(_classes.Count);
            // ENode node = new ENode(expr, eclass, children);

            // node.AddUsed();
            // eclass.Nodes.Add(node);

            // _nodes.Add(expr, node);
            // _classes.Add(eclass);

            // return;
            return new EClass(_classes.Count); //new ENode(expr, children);
        }


        public bool Merge(EClass to, EClass from)
        {
            to = to.Find();
            from = to.Find();
            if (to == from)
            {
                return false;
            }

            _version++;

            to.Used.AddRange(from.Used);
            from.Used.Clear();

            to.Nodes.AddRange(from.Nodes);
            from.Nodes.Clear();

            _worklist.Add(to);
            return true;
        }

        public void ReBuild()
        {
            while (_worklist.Count > 0)
            {
                // remove same eclass avoid duplicate repair
                var todos = _worklist.GroupBy(x => x).Select(x => x.First());
                _worklist.Clear();
                foreach (var eclass in todos)
                {
                    Repair(eclass);
                }
            }
        }

        public void Repair(EClass eclass)
        {
            // copy and reset the used, will reassgin new used
            var used = new List<(ENode, EClass)>(eclass.Used);
            eclass.Used.Clear();
            foreach (var (pnode, pclass) in used)
            {   // update the parent node.
                if (_hascons.ContainsKey(pnode))
                {
                    pclass.Nodes.Remove(pnode);
                    _hascons.Remove(pnode);
                }
                var newPnode = pnode.Canonicalize();
                _hascons.Add(newPnode, pclass.Find());
                pclass.Find().Nodes.Add(newPnode);
            }
            /* when merged eclasses, some enodes will be same. 
            eg. b1 = 1 + mul(a,2) and b2 = 1 + a<<2. =>  b1 will equal b2
            so we need remove duplicate enode b1 and b2.  */
            var newUsed = new Dictionary<ENode, EClass>();
            foreach (var (pnode, pclass) in used)
            {
                var newPnode = pnode.Canonicalize();
                if (newUsed.ContainsKey(newPnode))
                {
                    Merge(pclass, newUsed[newPnode]);
                }
                newUsed.Add(newPnode, _hascons[newPnode]);
            }
            // reassgin current eclass's Used.
            foreach (var item in newUsed)
            {
                eclass.Find().Used.Add((item.Key, item.Value));
            }
        }


        private sealed class ENodeConverter : ExprVisitor<EClass, IRType>
        {
            private readonly EGraph _graph;

            public ENodeConverter(EGraph graph)
            {
                _graph = graph;
            }

            public override EClass VisitLeaf(Call expr)
            {
                var children = new[] { Visit(expr.Target) }.Concat(
                    from p in expr.Parameters select Visit(p)).ToArray();
                return _graph.MakeNode(expr, children);
            }

            public override EClass VisitLeaf(Const expr)
            {
                return _graph.MakeNode(expr, Array.Empty<EClass>());
            }

            public override EClass VisitLeaf(Function expr)
            {
                var children = (from p in expr.Parameters select Visit(p))
                    .Concat(new[] { Visit(expr.Body) }).ToArray();
                return _graph.MakeNode(expr, children);
            }

            public override EClass VisitLeaf(IR.Tuple expr)
            {
                var children = (from p in expr.Fields select Visit(p)).ToArray();
                return _graph.MakeNode(expr, children);
            }

            public override EClass VisitLeaf(Op expr)
            {
                return _graph.MakeNode(expr, Array.Empty<EClass>());
            }

            public override EClass VisitLeaf(Var expr)
            {
                return _graph.MakeNode(expr, Array.Empty<EClass>());
            }
        }
    }
}
