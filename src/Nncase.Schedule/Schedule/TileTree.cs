// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections;
using Nncase.IR;
using VisitorPatternGenerator;

namespace Nncase.Schedule;


public abstract partial class TileTreeNode
{
    private readonly List<TileTreeNode> _children;

    public TileTreeNode(TileTreeNode? parent = null)
    {
        Parent = parent;
        _children = new();
    }

    public TileTreeNode? Parent { get; private set; }

    public IList<TileTreeNode> Children => _children;

    public void Add(TileTreeNode node)
    {
        node.Parent = this;
        _children.Add(node);
    }

    public void Remove(TileTreeNode node)
    {
        _children.Remove(node);
    }
}

[Acceptor<TileTreeNode>]
public partial class ScopeNode
{
    public ScopeNode()
    {
    }
}

[Acceptor<TileTreeNode>]
public partial class TileNode
{
    public TileNode()
    {
    }
}

[Visitor<TileTreeNode>]
public partial interface ITreeNodeVisitor { }

public static class TreeSearch
{
    public static void Search()
    {
        // three op
        // matmul
        // exp
        // matmul
        var tree = new ScopeNode();
        // tree.Accept()
        // tree.Add(new TileNode());
        // tree.Add(new TileNode());
        // tree.Add(new TileNode());
    }
}
