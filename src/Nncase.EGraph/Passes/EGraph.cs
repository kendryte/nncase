// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Reactive;
using System.Runtime.CompilerServices;
using NetFabric.Hyperlinq;
using Nncase.IR;

namespace Nncase.Passes;

/// <summary>
/// EGraph.
/// </summary>
public sealed partial class EGraph : IEGraph
{
    // TODO: use weak keys
    private readonly Dictionary<ENode, ENodeEntry> _nodes = new();
    private readonly List<EClass> _classes = new();

    /// <summary>
    /// which eclass should be repair.
    /// </summary>
    private readonly Queue<WorkItem> _worklist = new();

    private int _version;
    private int _globalEClassId;

    /// <summary>
    /// Initializes a new instance of the <see cref="EGraph"/> class.
    /// </summary>
    public EGraph()
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="EGraph"/> class.
    /// </summary>
    /// <param name="expr">Root expression.</param>
    public EGraph(Expr expr)
    {
        Root = Add(expr);
    }

    public EClass? Root { get; set; }

    /// <inheritdoc/>
    public IEnumerable<EClass> Classes => _classes;

    /// <inheritdoc/>
    public IEnumerable<ENode> Nodes => _nodes.Keys;

    /// <inheritdoc/>
    public int Version => _version;

    /// <inheritdoc/>
    public EClass Add(Expr expr)
    {
        if (expr.CheckedType is null)
        {
            expr.InferenceType();
        }

        var converter = new ENodeConverter(this);
        return converter.Visit(expr);
    }

    /// <inheritdoc/>
    public EClass Find(ENode node)
    {
        return _nodes[node].Class;
    }

    /// <inheritdoc/>
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

        // choice the more accurate checked type
        switch (classA.CheckedType.CompareTo(classB.CheckedType))
        {
            case < 0:
                classA.SetCheckedType(classB.CheckedType);
                break;
            case > 0:
                classB.SetCheckedType(classA.CheckedType);
                break;
        }

        _worklist.Enqueue(new() { OldClass = classB, NewClass = classA });
        return true;
    }

    /// <inheritdoc/>
    public void Rebuild()
    {
        while (_worklist.Count > 0)
        {
            Repair(_worklist.Dequeue());
        }
    }

    public EClass AddENode(Expr expr, IRArray<EClass> children)
    {
        // TODO: concurrent safe
        EClass eclass;

        // 2. Create new node and check it
        var enode = ENode.Create(expr, children);
        if (_nodes.TryGetValue(enode, out var entry))
        {
            // 2.1 Node already exists, add expr to memo
            eclass = entry.Class;
        }
        else
        {
            // 3. New node and new class
            eclass = new EClass(_globalEClassId++);
            entry = new ENodeEntry(enode, eclass);
            _classes.Add(eclass);
            _nodes.Add(enode, entry);
            enode.SetClass(eclass);
        }

        // 4. When enode is new created, maybe we can get the more accurate ir type.
        if (expr.CheckedType?.CompareTo(eclass.CheckedType) > 0)
        {
            eclass.SetCheckedType(expr.CheckedType);
        }

        return eclass;
    }

    private void Repair(WorkItem workItem)
    {
        if (workItem.OldClass.Id == 29)
        {
        }

        workItem.NewClass = workItem.NewClass.Find();
        foreach (var enode in workItem.OldClass.UsedBy)
        {
            // 1. Check this node is alive and remove it
            if (_nodes.TryGetValue(enode, out var originalEntry))
            {
                var originalClass = originalEntry.Class;

                _nodes.Remove(enode);
                originalClass.RemoveNode(enode);

                // 2. Replace node's children
                var newNode = enode.Canonicalize();

                if (_nodes.TryGetValue(newNode, out var existingEntry))
                {
                    // 3. If new node already exists, update memo and union these classes
                    originalEntry.Node = existingEntry.Node;
                    originalEntry.Class = existingEntry.Class;

                    Union(originalClass, existingEntry.Class);
                }
                else
                {
                    // 4. Add new node
                    originalEntry.Node = newNode;
                    _nodes.Add(newNode, originalEntry);

                    newNode.SetClass(originalClass);
                }
            }
        }

        foreach (var enode in workItem.OldClass.Nodes)
        {
            if (_nodes.TryGetValue(enode, out var entry))
            {
                entry.Class = workItem.NewClass;
                workItem.NewClass.AddNode(entry.Node);
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

    private class ENodeEntry
    {
        public ENodeEntry(ENode node, EClass @class)
        {
            Node = node;
            Class = @class;
        }

        public ENode Node { get; set; }

        public EClass Class { get; set; }

        public override string ToString()
        {
            return $"{Node}, {Class}";
        }
    }

    private sealed class ENodeConverter : ExprVisitor<EClass, Unit>
    {
        private readonly EGraph _graph;

        public ENodeConverter(EGraph graph)
        {
            _graph = graph;
        }

        protected override EClass DefaultVisitLeaf(Expr expr)
        {
            EClass[]? operands;
            if (expr is BaseFunction baseFunction && !CanVisitFunctionBody(baseFunction))
            {
                operands = Array.Empty<EClass>();
            }
            else
            {
                operands = expr.Operands.AsValueEnumerable().Select(x => ExprMemo[x]).ToArray();
            }

            return _graph.AddENode(expr, operands);
        }
    }
}
