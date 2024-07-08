// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections;
using System.Reactive;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Affine;
using VisitorPatternGenerator;

namespace Nncase.Schedule.TileTree;

public partial class TreeWalker : ITreeNodeVisitor<Unit, Unit>
{
    public TreeWalker()
    {
        Nodes = new();
    }

    public List<ITreeNode> Nodes { get; }

    public static List<ITreeNode> Walk(ITreeNode tree)
    {
        var x = new TreeWalker();
        tree.Accept(x, default);
        return x.Nodes;
    }

    public Unit Visit(ScopeNode value, Unit arg1)
    {
        Nodes.Add(value);
        foreach (var c in value.Children)
        {
            c.Accept(this, arg1);
        }

        return default;
    }

    public Unit Visit(TileNode value, Unit arg1)
    {
        Nodes.Add(value);
        return value.Child.Accept(this, arg1);
    }

    public Unit Visit(OpNode value, Unit arg1)
    {
        Nodes.Add(value);
        return default;
    }
}
