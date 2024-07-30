// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Reactive;
using VisitorPatternGenerator;

namespace Nncase.Schedule.TileTree;

public sealed record MergePoint(int Consumer, int Producer, int Level)
{
    public override string ToString() => $"merge({Consumer},{Producer},{Level})";
}

internal sealed class TreeMergePointCollector : ITreeNodeVisitor<Unit, Unit>
{
    public TreeMergePointCollector(int targetLevel)
    {
        TargetLevel = targetLevel;
        Points = new();
    }

    public int TargetLevel { get; }

    public List<MergePoint> Points { get; }

    public Unit Visit(ScopeNode value, Unit arg1)
    {
        if (value.Children.All(x => x is TileNode node && node.Level == TargetLevel))
        {
            for (int consumer = value.Children.Count - 1; consumer > 0; consumer--)
            {
                Points.Add(new MergePoint(((TileNode)value.Children[consumer]).OpId, ((TileNode)value.Children[consumer - 1]).OpId, TargetLevel));
            }
        }

        foreach (var item in value.Children)
        {
            item.Accept(this, arg1);
        }

        return default;
    }

    public Unit Visit(TileNode value, Unit arg1)
    {
        return value.Child.Accept(this, arg1);
    }

    public Unit Visit(OpNode value, Unit arg1)
    {
        return default;
    }
}
