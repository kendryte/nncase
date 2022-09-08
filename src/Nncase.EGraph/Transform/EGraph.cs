﻿// Copyright (c) Canaan Inc. All rights reserved.
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
public sealed partial class EGraph : IEGraph
{
    private readonly Dictionary<Expr, ENode> _exprMemo = new();
    private readonly Dictionary<ENode, EClass> _nodes = new();
    private readonly List<EClass> _classes = new();

    private int _version = 0;
    private int _globalEClassId = 0;

    /// <summary>
    /// which eclass should be repair.
    /// </summary>
    private Queue<WorkItem> _worklist = new();

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
    /// <see cref="_hascons"/>.
    /// </summary>
    public IEnumerable<EClass> Classes => _classes;

    /// <inheritdoc/>
    public IEnumerable<ENode> Nodes => _nodes.Keys;

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
    /// Find eclass of enode.
    /// </summary>
    /// <param name="node">ENode.</param>
    /// <returns>EClass.</returns>
    public EClass Find(ENode node)
    {
        return _nodes[node].Find();
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
        if (classA.Nodes.Count < classB.Nodes.Count)
        {
            (classA, classB) = (classB, classA);
        }

        classB.Parent = classA;
        classA.AddNodes(classB.Nodes);
        _worklist.Enqueue(new() { OldClass = classB, NewClass = classA });
        return true;
    }

    /// <summary>
    /// After merge, we use rebuild get new dep information.
    /// </summary>
    public void Rebuild()
    {
        while (_worklist.Count > 0)
        {
            Repair(_worklist.Dequeue());
        }
    }

    private EClass AddENode(Expr expr, IRArray<EClass> children)
    {
        // TODO: concurrent safe
        if (!_exprMemo.TryGetValue(expr, out var enode))
        {
            enode = new ENode(expr, children);
            _exprMemo.Add(expr, enode);
        }

        if (!_nodes.TryGetValue(enode, out var eclass))
        {
            eclass = new EClass(_globalEClassId++);
            _classes.Add(eclass);
            enode.AddUsed(eclass);
            _nodes.Add(enode, eclass);
        }

        return eclass;
    }

    private void Repair(WorkItem workItem)
    {
        workItem.NewClass = workItem.NewClass.Find();

        foreach (var enode in workItem.OldClass.Used)
        {
            // 1. Check this node is alive and remove it
            if (_nodes.TryGetValue(enode, out var originalClass))
            {
                originalClass = originalClass.Find();
                _nodes.Remove(enode);
                _exprMemo.Remove(enode.Expr);
            }
            else
            {
                continue;
            }

            // 2. Update node's children
            var newNode = enode.Canonicalize();
            originalClass.RemoveNode(enode);

            // 3. If new node already exists, union these classes.
            if (_nodes.TryGetValue(newNode, out var existsClass))
            {
                Union(originalClass, existsClass);
            }
            else
            {
                newNode.AddUsed(originalClass);
                _nodes.Add(newNode, originalClass);
            }
        }

        foreach (var enode in workItem.OldClass.Nodes)
        {
            if (_nodes.ContainsKey(enode))
            {
                _nodes[enode] = workItem.NewClass;
            }
            else
            {
                workItem.NewClass.RemoveNode(enode);
            }
        }

        workItem.OldClass.Kill();
        _classes.Remove(workItem.OldClass);
    }

    private struct WorkItem
    {
        public EClass OldClass;
        public EClass NewClass;
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

        public override EClass VisitLeaf(Marker expr)
        {
            var children = new[] { ExpressionMemo[expr.Target], ExpressionMemo[expr.Attribute] };
            return _graph.AddENode(expr, children);
        }

        public override EClass VisitLeaf(None expr)
        {
            return _graph.AddENode(expr, Array.Empty<EClass>());
        }
    }
}
