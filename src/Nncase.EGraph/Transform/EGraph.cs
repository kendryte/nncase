// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using NetFabric.Hyperlinq;
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

        /// <summary>
        /// NOTE the Used mean which Enode use this EClass. eg. z = x + y. the EClass's Used will add {(z, z's eclass id)}.
        /// <remark> It's Not mean this EClass's Nodes </remark>
        /// </summary>
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
        /// <summary>
        /// record each Enode's Eclass
        /// </summary>
        private readonly Dictionary<ENode, EClass> _hascons = new Dictionary<ENode, EClass>();

        private readonly List<EClass> _classes = new List<EClass>();
        private int _version = 0;

        private List<EClass> _worklist = new List<EClass>();

        /// <summary>
        /// the all EClass and it's 
        /// </summary>
        /// <returns></returns>
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

        /// <summary>
        /// <see cref="_hascons"/>
        /// </summary>
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

        /// <summary>
        /// merge two equal Eclass
        /// </summary>
        /// <param name="to"></param>
        /// <param name="from"></param>
        /// <returns></returns>
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

        /// <summary>
        /// After merge, we use rebuild get new dep information
        /// </summary>
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


        /// <summary>
        /// <see cref="TopSort(Dictionary{EClass, List{ENode}})"/>
        /// </summary>
        /// <returns></returns>
        public EClass[] TopSort() => TopSort(EClasses());

        /// <summary>
        /// Get Top Sorted EGraph Datastruce.
        /// </summary>
        /// <returns> the Eclass array, the root eclass is first one </returns>
        public EClass[] TopSort(Dictionary<EClass, List<ENode>> eClasses)
        {
            void dfs(EClass eclass, Dictionary<EClass, bool> visited, List<EClass> paths)
            {
                visited[eclass] = true;
                foreach (var (_, used_eclass) in eclass.Used)
                {
                    if (!visited[eclass])
                        dfs(used_eclass, visited, paths);
                }
                paths.Add(eclass); // put the root node into first
            }

            var visited = new Dictionary<EClass, bool>();
            foreach (var (k, _) in eClasses)
            {
                visited[k] = false;
            }
            var paths = new List<EClass>();
            foreach (var (eclass, is_visited) in visited)
            {
                if (!is_visited)
                {
                    dfs(eclass, visited, paths);
                }
            }
            return paths.ToArray();
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
                var children = new[] { ExpressionMemo[expr.Target] }.Concat(
                    from p in expr.Parameters select ExpressionMemo[p]).ToArray();
                return _graph.AddENode(expr, children);
            }

            public override EClass VisitLeaf(Const expr)
            {
                return _graph.AddENode(expr, Array.Empty<EClass>());
            }

            public override EClass VisitLeaf(Function expr)
            {
                var children = new[] { ExpressionMemo[expr.Body] }.Concat(
                  from p in expr.Parameters select ExpressionMemo[p]).ToArray();
                return _graph.AddENode(expr, children);
            }

            public override EClass VisitLeaf(IR.Tuple expr)
            {
                var children = (from p in expr.Fields select ExpressionMemo[p]).ToArray();
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
