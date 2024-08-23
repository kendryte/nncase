// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections;
using System.Reactive;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Affine;

namespace Nncase.Schedule.TileGraph;

public partial class GraphWalker
{
    private GraphWalker()
    {
        Graphs = new();
    }

    public List<TieredTileGraph> Graphs { get; }

    public static List<TieredTileGraph> Walk(TieredTileGraph tree)
    {
        var walker = new GraphWalker();
        walker.Visit(tree, default);
        return walker.Graphs;
    }

    public Unit Visit(TieredTileGraph value, Unit arg1)
    {
        Graphs.Add(value);
        foreach (var c in value.Clusters.OfType<TieredTileGraph>())
        {
            Visit(c, arg1);
        }

        return default;
    }
}
