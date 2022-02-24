// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.IR;

namespace Nncase.Transform;

/// <summary>
/// EGraph.
/// </summary>
public sealed partial class EGraph
{
    private readonly Dictionary<Expr, ENode> _exprMemo = new();
    private readonly Dictionary<ENode, EClass> _nodes = new();

    /// <summary>
    /// record each Enode's Eclass.
    /// </summary>
    private readonly Dictionary<ENode, EClass> _hascons = new Dictionary<ENode, EClass>();

    // private readonly List<EClass> _classes = new List<EClass>();
    private int _version = 0;

    private int _globalEclassId = 0;

    /// <summary>
    /// save which node has been merged,
    /// we should update it's eclass in hashcon.
    /// </summary>
    private readonly List<ENode> _mergedlist = new();

    /// <summary>
    /// which eclass should be repair.
    /// </summary>
    private List<EClass> _worklist = new();

    /// <summary>
    /// the all EClass and it's.
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
            {
                eclasses.Add(parentEclass, new List<ENode> { enode });
            }
            else
            {
                eclasses[parentEclass].Add(enode);
            }
        }

        // foreach (var enode in _mergedlist) { _hascons[enode] = _hascons[enode].Find(); }
        // _mergedlist.Clear();
        return eclasses;
    }

    /// <summary>
    /// <see cref="_hascons"/>.
    /// </summary>
    public IReadOnlyDictionary<ENode, EClass> HashCons => _hascons;

    /// <summary>
    /// Initializes a new instance of the <see cref="EGraph"/> class.
    /// </summary>
    public EGraph()
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="EGraph"/> class.
    /// </summary>
    /// <param name="expr">Expression.</param>
    public EGraph(Expr expr)
    {
        Add(expr);
    }

    /// <summary>
    /// Gets version.
    /// </summary>
    public int Version => _version;

    /// <summary>
    /// Add expr, get the eclass id.
    /// </summary>
    /// <param name="expr">Expression.</param>
    /// <returns>Eclass of this node.</returns>
    public EClass Add(Expr expr)
    {
        if (expr.CheckedType is null)
        {
            expr.InferenceType();
        }

        var converter = new ENodeConverter(this);
        return converter.Visit(expr);
    }

    /// <summary>
    /// Union two equal Eclass.
    /// </summary>
    /// <param name="classA">class a.</param>
    /// <param name="classB">class b.</param>
    /// <returns>If version changed.</returns>
    public bool Union(EClass classA, EClass classB)
    {
        classA = classA.Find();
        classB = classB.Find();
        if (classA == classB)
        {
            return false;
        }

        _version++;
        if (classA.Used.Count < classB.Used.Count)
        {
            (classA, classB) = (classB, classA);
        }

        classB.Parent = classA;
        classA.Used.AddRange(classB.Used);
        classB.Used.Clear();

        _worklist.Add(classA);
        return true;
    }

    /// <summary>
    /// After merge, we use rebuild get new dep information.
    /// </summary>
    public void Rebuild()
    {
        while (_worklist.Count > 0)
        {
            // swap todos and worklist
            (var todos, _worklist) = (_worklist, new());

            foreach (var eclass in todos)
            {
                Repair(eclass);
            }
        }
    }

    /// <summary>
    /// <see cref="TopSort(IReadOnlyDictionary&lt;EClass, List&lt;ENode&gt;&gt;)"/>.
    /// </summary>
    /// <returns></returns>
    public EClass[] TopSort() => TopSort(EClasses());

    /// <summary>
    /// Get Top Sorted EGraph Datastruce.
    /// </summary>
    /// <returns> the Eclass array, the root eclass is first one. </returns>
    public EClass[] TopSort(IReadOnlyDictionary<EClass, List<ENode>> eClasses)
    {
        void dfs(EClass eclass, Dictionary<EClass, bool> visited, List<EClass> paths)
        {
            visited[eclass] = true;
            foreach (var (_, used_eclass) in eclass.Used)
            {
                if (!visited[eclass])
                {
                    dfs(used_eclass, visited, paths);
                }
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

    private EClass AddENode(Expr expr, IRArray<EClass> children)
    {
        if (!_exprMemo.TryGetValue(expr, out var enode))
        {
            enode = new ENode(expr, children);
        }

        if (!_nodes.TryGetValue(enode, out var eclass))
        {
            eclass = new EClass(_globalEclassId++);
            enode.AddUsed(eclass);
        }

        return eclass.Find();
    }

    private void Repair(EClass eclass)
    {
        // copy and reset the used, will reassgin new used
        var oldUsed = new List<(ENode, EClass)>(eclass.Used);
        eclass.Used.Clear();

        foreach (var (pnode, pclass) in oldUsed)
        {
            // update the parent node.
            if (_hascons.ContainsKey(pnode))
            {
                _hascons.Remove(pnode);
            }

            // TODO we update the enode, should put this new node to it's child eclass's oldUsed.
            // Then when that eclass be repaired, it will update this new enode.
            var (newPnode, newParents) = pnode.Canonicalize(eclass);
            var newPclass = pclass.Find();
            if (!_hascons.TryGetValue(newPnode, out var result))
            {
                _hascons.Add(newPnode, newPclass); // update this node to it's child's used
                newParents.ForEach(parent => parent.Used.Add((newPnode, newPclass)));
            }
            else if (result.Find() != newPclass)
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
                Union(pclass, newUsed[newPnode]);
            }

            newUsed[newPnode] = _hascons[newPnode];
        }

        // reassgin current eclass's Used.
        eclass.Find().Used.AddRange(newUsed.Select(kv => (kv.Key, kv.Value)));
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
