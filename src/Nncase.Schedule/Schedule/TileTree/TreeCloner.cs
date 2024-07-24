// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Reactive;
using VisitorPatternGenerator;

namespace Nncase.Schedule.TileTree;

internal partial class TreeCloner : ITreeNodeVisitor<Unit, ITreeNode>
{
    private readonly Dictionary<ITreeNode, ITreeNode> _memo = new(ReferenceEqualityComparer.Instance);

    public ITreeNode Visit(ScopeNode value, Unit arg1)
    {
        if (!_memo.TryGetValue(value, out var nScope))
        {
            nScope = new ScopeNode();
            foreach (var item in value.Children)
            {
                ((ScopeNode)nScope).Add(item.Accept(this, arg1));
            }

            _memo.Add(value, nScope);
        }

        return nScope;
    }

    public ITreeNode Visit(TileNode value, Unit arg1)
    {
        if (!_memo.TryGetValue(value, out var nTile))
        {
            nTile = new TileNode(value.Level, value.OpId, value.DimNames)
            {
                DomainRelation = value.DomainRelation,
                Child = value.Child.Accept(this, arg1),
            };
            _memo.Add(value, nTile);
        }

        return nTile;
    }

    public ITreeNode Visit(OpNode value, Unit arg1)
    {
        if (!_memo.TryGetValue(value, out var nOp))
        {
            nOp = new OpNode(value.Grid, value.Op, value.OpId, value.DimNames, value.DomainBounds, value.BufferShapes.Select(x => (IEnumerable<int>)x), value.Dependences.Select(d => new OpNode.Dependence(d.Index, (OpNode)_memo[d.Node])))
            {
                DomainRelation = value.DomainRelation,
            };

            _memo.Add(value, nOp);
        }

        return nOp;
    }
}
