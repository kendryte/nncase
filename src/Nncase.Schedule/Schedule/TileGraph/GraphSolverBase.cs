// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Google.OrTools.ConstraintSolver;
using Nncase.IR.Affine;

namespace Nncase.Schedule.TileGraph;

public interface ITileGraphVisitor<TArg, TReturn>
{
    TReturn Visit(TileGraph value, TArg arg1);

    TReturn Visit(OpNode value, TArg arg1);
}

public abstract class GraphSolverBase
{
    public GraphSolverBase(Solver solver, Dictionary<OpNode, OpNodeInfo> opNodeMemo, Dictionary<TileGraph, TileNodeInfo> tileNodeMemo, Dictionary<ITileableNode, DomainInfo> tileableNodeMemo, ITargetOptions targetOptions)
    {
        Solver = solver;
        OpNodeMemo = opNodeMemo;
        TileNodeMemo = tileNodeMemo;
        TileableNodeMemo = tileableNodeMemo;
        TargetOptions = targetOptions;
    }

    public Solver Solver { get; }

    public Dictionary<OpNode, OpNodeInfo> OpNodeMemo { get; }

    public Dictionary<TileGraph, TileNodeInfo> TileNodeMemo { get; }

    public Dictionary<ITileableNode, DomainInfo> TileableNodeMemo { get; }

    public ITargetOptions TargetOptions { get; }

    /// <summary>
    /// compute the dim map from the domain relation.
    /// </summary>
    /// <param name="value">node.</param>
    /// <returns>map.</returns>
    public static Dictionary<int, int> GetDimsMap(ITileableNode value)
    {
        var map = new Dictionary<int, int>();
        var relation = value.DomainRelation.Map;
        for (int i = 0; i < relation.Results.Length; i++)
        {
            if (relation.Results[i] is { Offset: AffineDim dim, Extent: AffineExtent ext } && dim.Position == ext.Position)
            {
                map[i] = dim.Position;
            }
        }

        return map;
    }
}
