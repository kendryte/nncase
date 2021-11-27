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
                children.Used.Add((this, eClass.Find()));
            }
        }

        public ENode Canonicalize()
        {
            var children = (from c in Children select c.Find()).ToArray();
            return new ENode(Expr, children);
        }

        public (ENode, List<EClass>) Canonicalize(EClass TargeteClass)
        {
            var todos = new List<EClass>();
            EClass find_other_parents(EClass child)
            {
                var neweClass = child.Find();
                if (neweClass != TargeteClass)
                    todos.Add(neweClass);
                return neweClass;
            }
            return (new ENode(Expr, Children.Select(find_other_parents).ToArray()), todos);
        }

        public override string ToString()
        {
            var str = String.Join(", ", Children.Select(x => x.Id));
            return $"{Expr.GetType().Name} ({str})";
        }
    }

    /// <summary>
    /// EClass.
    /// </summary>
    public sealed partial class EClass
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

        public override string ToString() => $"{Id} -> {Parent?.Id}";
    }

    /// <summary>
    /// EGraph.
    /// </summary>
    public sealed partial class EGraph
    {
        /// <summary>
        /// record each Enode's Eclass
        /// </summary>
        private readonly Dictionary<ENode, EClass> _hascons = new Dictionary<ENode, EClass>();

        // private readonly List<EClass> _classes = new List<EClass>();
        private int _version = 0;

        private int _globalEclassId = 0;

        /// <summary>
        /// save which node has been merged,
        /// we should update it's eclass in hashcon;
        /// </summary>
        private readonly List<ENode> _mergedlist = new();

        /// <summary>
        /// which eclass should be repair
        /// </summary>
        private readonly List<EClass> _worklist = new();

        /// <summary>
        /// the all EClass and it's 
        /// </summary>
        /// <returns></returns>
        public Dictionary<EClass, List<ENode>> EClasses()
        {
            var eclasses = new Dictionary<EClass, List<ENode>>();
            foreach (var (enode, eclass) in _hascons)
            {
                // NOTE when do Find in here, maybe some enode's child id haven't update.
                var parentEclass = eclass.Find();
                // if (parentEclass != eclass) { _mergedlist.Add(enode); }
                if (!eclasses.ContainsKey(parentEclass))
                    eclasses.Add(parentEclass, new List<ENode> { enode });
                else
                    eclasses[parentEclass].Add(enode);
            }
            // foreach (var enode in _mergedlist) { _hascons[enode] = _hascons[enode].Find(); }
            // _mergedlist.Clear();
            return eclasses;
        }

        /// <summary>
        /// <see cref="_hascons"/>
        /// </summary>
        public IReadOnlyDictionary<ENode, EClass> HashCons => _hascons;

        public int Version => _version;


        public EGraph() { }

        public EGraph(Expr expr) => Add(expr, out var eClass);

        /// <summary>
        /// add expr, get the eclass id
        /// </summary>
        /// <param name="expr"></param>
        /// <param name="eClass"></param>
        public void Add(Expr expr, out EClass eClass)
        {
            var converter = new ENodeConverter(this);
            eClass = converter.Visit(expr);
        }

        /// <summary>
        /// <see cref="Add(Expr, out EClass)"/>
        /// </summary>
        /// <param name="expr"></param>
        public void Add(Expr expr) => Add(expr, out var eClass);

        public void Clear()
        {
            _worklist.Clear();
            _version = 0;
            _hascons.Clear();
            // _classes.Clear();
        }

        private EClass AddENode(Expr expr, IRArray<EClass> children)
        {
            // todo rewrite it.
            ENode enode = new ENode(expr, children);
            if (!_hascons.TryGetValue(enode, out var eclass))
            {
                eclass = new EClass(_globalEclassId++);
                enode.AddUsed(eclass);
                _hascons.Add(enode, eclass);
            }
            return eclass.Find();
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
                var todos = _worklist.Select(x => x.Find()).Distinct().ToArray();
                _worklist.Clear();
                foreach (var eclass in todos)
                {
                    RePair(eclass);
                }
            }
        }

        private void RePair(EClass eclass)
        {
            // copy and reset the used, will reassgin new used
            var oldUsed = new List<(ENode, EClass)>(eclass.Used);
            eclass.Used.Clear();
            foreach (var (pnode, pclass) in oldUsed)
            {   // update the parent node.
                if (_hascons.ContainsKey(pnode))
                    _hascons.Remove(pnode);
                // TODO we update the enode, should put this new node to it's child eclass's oldUsed. 
                // Then when that eclass be repaired, it will update this new enode.  
                var (newPnode, newParents) = pnode.Canonicalize(eclass);
                var newPclass = pclass.Find();
                if (!_hascons.TryGetValue(newPnode, out var result))
                {
                    _hascons.Add(newPnode, newPclass); // update this node to it's child's used 
                    newParents.ForEach(parent => parent.Used.Add((newPnode, newPclass)));
                }
                else if (result != newPclass)
                {
                    throw new InvalidProgramException($"The ENode {newPnode}'s Eclass is {_hascons[newPnode]} but will set to {newPclass}!");
                }
            }
            /* Eg. [(x*2)+1] and [(x<<1)+1], when (x*2) <== (x<<1), 
               the [(x*2)+1] and [(x<<1)+1] will got same new enode,
                so we should merge them.
             */
            var newUsed = new Dictionary<ENode, EClass>();
            foreach (var (pnode, pclass) in oldUsed)
            {
                var newPnode = pnode.Canonicalize();
                if (newUsed.ContainsKey(newPnode))
                {
                    Merge(pclass, newUsed[newPnode]);
                }
                newUsed[newPnode] = _hascons[newPnode];
            }
            // reassgin current eclass's Used.
            eclass.Find().Used.AddRange(newUsed.Select(kv => (kv.Key, kv.Value)));
        }


        /// <summary>
        /// <see cref="TopSort(IReadOnlyDictionary<EClass, List<ENode>>)"/>
        /// </summary>
        /// <returns></returns>
        public EClass[] TopSort() => TopSort(EClasses());

        /// <summary>
        /// Get Top Sorted EGraph Datastruce.
        /// </summary>
        /// <returns> the Eclass array, the root eclass is first one </returns>
        public EClass[] TopSort(IReadOnlyDictionary<EClass, List<ENode>> eClasses)
        {
            void dfs(EClass eclass, Dictionary<EClass, bool> visited, List<EClass> paths)
            {
                visited[eclass] = true;
                foreach (var (_, used_eclass) in eclass.Used)
                {
                    if (!visited[eclass])
                        dfs(used_eclass, visited, paths);
                }
                paths.Add(eclass); // put the root node into last
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
