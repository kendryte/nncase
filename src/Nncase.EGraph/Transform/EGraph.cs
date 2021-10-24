// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.IR;


namespace Nncase.Transform
{
    /// <summary>
    /// ENode.
    /// </summary>
    public sealed record ENode(Expr Expr, IRArray<EClass> Children)
    {
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

        public EClass? Parent = null;

        public readonly List<(ENode, EClass)> Used = new List<(ENode, EClass)>();

        public EClass Find()
        {
            if (Parent is null)
            {
                return this;
            }
            Parent = Parent.Find();
            return Parent;
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

        private List<EClass> _worklist = new List<EClass>();

        public Dictionary<EClass, List<ENode>> EClasses()
        {
            var eclasses = new Dictionary<EClass, List<ENode>>();
            foreach (var (enode, eclass) in _hascons)
            {
                var parentEclass = eclass.Find();
                if (!eclasses.ContainsKey(parentEclass))
                    eclasses.Add(parentEclass, new List<ENode> { enode });
                else
                    eclasses[parentEclass].Add(enode);
            }
            return eclasses;
        }
        public Dictionary<ENode, EClass> Nodes => _hascons;

        public int Version => _version;


        public EGraph() { }

        public EGraph(Expr expr) => Add(expr);

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

        public void Clear()
        {
            _worklist.Clear();
            _version = 0;
            _hascons.Clear();
            _classes.Clear();
        }

        private EClass AddENode(Expr expr, IRArray<EClass> children)
        {
            // todo rewrite it.
            ENode enode = new ENode(expr, children);
            if (!_hascons.TryGetValue(enode, out var eclass))
            {
                eclass = new EClass(_classes.Count);
                enode.AddUsed(eclass);

                _hascons.Add(enode, eclass);
                _classes.Add(eclass);
            }
            return eclass;
        }


        public bool Merge(EClass to, EClass from)
        {
            to = to.Find();
            from = from.Find();
            if (to == from)
            {
                return false;
            }

            _version++;
            from.Parent = to;

            to.Used.AddRange(from.Used);
            from.Used.Clear();

            _worklist.Add(to);
            return true;
        }

        public void ReBuild()
        {
            while (_worklist.Count > 0)
            {
                // remove same eclass avoid duplicate repair
                var todos = _worklist.Select(x => x.Find()).Distinct();
                _worklist = new List<EClass>();
                foreach (var eclass in todos)
                {
                    RePair(eclass);
                }
            }
        }

        private void RePair(EClass eclass)
        {
            // copy and reset the used, will reassgin new used
            var used = new List<(ENode, EClass)>(eclass.Used);
            eclass.Used.Clear();
            foreach (var (pnode, pclass) in used)
            {   // update the parent node.
                if (_hascons.ContainsKey(pnode))
                {
                    _hascons.Remove(pnode);
                }
                var newPnode = pnode.Canonicalize();
                var newPclass = pclass.Find();
                _hascons.Add(newPnode, newPclass);
            }
            /* when merge eclass, eg. x+0 => x, the x's class will have the Node (x+0), then update the Node (x+0)ˆ, the newPclass will have (x+0) and (x+0)ˆ  */
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
                return _graph.AddENode(expr, children);
            }

            public override EClass VisitLeaf(Const expr)
            {
                return _graph.AddENode(expr, Array.Empty<EClass>());
            }

            public override EClass VisitLeaf(Function expr)
            {
                var children = (from p in expr.Parameters select Visit(p))
                    .Concat(new[] { Visit(expr.Body) }).ToArray();
                return _graph.AddENode(expr, children);
            }

            public override EClass VisitLeaf(IR.Tuple expr)
            {
                var children = (from p in expr.Fields select Visit(p)).ToArray();
                return _graph.AddENode(expr, children);
            }

            public override EClass VisitLeaf(Op expr)
            {
                return _graph.AddENode(expr, Array.Empty<EClass>());
            }

            public override EClass VisitLeaf(Var expr)
            {
                return _graph.AddENode(expr, Array.Empty<EClass>());
            }
        }
    }
}
