// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Reactive;

namespace Nncase.Schedule.TileTree;

internal partial class TreeFunctor : ITreeNodeVisitor<Unit, Unit>
{
    public TreeFunctor(Action<ITreeNode> func, bool preOrder)
    {
        Func = func;
        PreOrder = preOrder;
    }

    public Action<ITreeNode> Func { get; }

    public bool PreOrder { get; }

    public Unit Visit(ScopeNode value, Unit arg1)
    {
        if (PreOrder)
        {
            Func(value);
        }

        foreach (var c in value.Children)
        {
            c.Accept(this, arg1);
        }

        if (!PreOrder)
        {
            Func(value);
        }

        return default;
    }

    public Unit Visit(TileNode value, Unit arg1)
    {
        if (PreOrder)
        {
            Func(value);
        }

        value.Child.Accept(this, arg1);

        if (!PreOrder)
        {
            Func(value);
        }

        return default;
    }

    public Unit Visit(OpNode value, Unit arg1)
    {
        Func(value);
        return default;
    }
}
