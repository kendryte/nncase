// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Reactive;

namespace Nncase.Schedule.TileGraph;

internal sealed class GraphMergePointCollector
{
    public GraphMergePointCollector(int targetLevel)
    {
        TargetLevel = targetLevel;
        Points = new();
    }

    public int TargetLevel { get; }

    public List<MergePoint> Points { get; }

    public Unit Visit(TieredTileGraph value, Unit arg1)
    {
        var rootGraph = value.RootParent();
        var subgraphs = value.Clusters.OfType<TieredTileGraph>().Where(x => x.Level == TargetLevel);
        foreach (var s1 in subgraphs)
        {
            foreach (var s2 in subgraphs.Where(s => !ReferenceEquals(s, s1)))
            {
                foreach (var a in s1.Vertices)
                {
                    foreach (var b in s2.Vertices)
                    {
                        if (rootGraph.TryGetEdge(a, b, out _))
                        {
                            Points.Add(new MergePoint(b, a, TargetLevel));
                        }
                        else if (rootGraph.TryGetEdge(b, a, out _))
                        {
                            Points.Add(new MergePoint(a, b, TargetLevel));
                        }
                    }
                }
            }
        }

        if (value.ClustersCount != 0)
        {
            foreach (var child in value.Clusters.OfType<TieredTileGraph>())
            {
                Visit(child, arg1);
            }
        }
        else
        {
            foreach (var child in value.Vertices)
            {
                Visit(child, arg1);
            }
        }

        return default;
    }

    public Unit Visit(TileGrid value, Unit arg1)
    {
        return default;
    }
}
