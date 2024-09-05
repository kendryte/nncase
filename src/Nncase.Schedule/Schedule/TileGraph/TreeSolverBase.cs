// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Google.OrTools.ConstraintSolver;
using Nncase.IR.Affine;

namespace Nncase.Schedule.TileGraph;

public abstract class TreeSolverBase<T>
{
    public TreeSolverBase(Solver solver, Dictionary<OpNode, OpNodeInfo<T>> opNodeMemo, Dictionary<TileNode, TileNodeInfo<T>> tileNodeMemo, Dictionary<ITileable, DomainInfo<T>> tileableNodeMemo, ICpuTargetOptions targetOptions)
    {
        Solver = solver;
        OpNodeMemo = opNodeMemo;
        TileNodeMemo = tileNodeMemo;
        TileableNodeMemo = tileableNodeMemo;
        TargetOptions = targetOptions;
    }

    public Solver Solver { get; }

    public Dictionary<OpNode, OpNodeInfo<T>> OpNodeMemo { get; }

    public Dictionary<TileNode, TileNodeInfo<T>> TileNodeMemo { get; }

    public Dictionary<ITileable, DomainInfo<T>> TileableNodeMemo { get; }

    public ICpuTargetOptions TargetOptions { get; }

    /// <summary>
    /// compute the dim map from the domain relation.
    /// </summary>
    /// <param name="value">node.</param>
    /// <returns>map.</returns>
    public static Dictionary<int, int> GetDimsMap(ITileable value)
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
