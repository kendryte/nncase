// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Reactive;
using VisitorPatternGenerator;

namespace Nncase.Schedule.TileTree;

internal partial class TreeCloner : ITreeNodeVisitor<Unit, ITreeNode>
{
    public ITreeNode Visit(ScopeNode value, Unit arg1)
    {
        var newScope = new ScopeNode();
        foreach (var item in value.Children)
        {
            newScope.Add(item.Accept(this, arg1));
        }

        return newScope;
    }

    public ITreeNode Visit(TileNode value, Unit arg1)
    {
        var newTile = new TileNode(value.Level, value.OpId, value.DimNames)
        {
            DomainRelation = value.DomainRelation,
            Child = value.Child.Accept(this, arg1),
        };
        return newTile;
    }

    public ITreeNode Visit(OpNode value, Unit arg1)
    {
        return new OpNode(value.Grid, value.Op, value.OpId, value.DimNames, value.DomainBounds, value.BufferShapes.Select(x => (IEnumerable<int>)x), value.Dependences)
        {
            DomainRelation = value.DomainRelation,
        };
    }
}
