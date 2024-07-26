// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Google.OrTools.ConstraintSolver;

namespace Nncase.Schedule.TileTree;

public abstract class TreeSolverBase
{
    public TreeSolverBase(Solver solver, Dictionary<OpNode, OpNodeInfo> opNodeMemo, Dictionary<TileNode, TileNodeInfo> tileNodeMemo, Dictionary<ITileAbleNode, DomainInfo> tileableNodeMemo, ITargetOptions targetOptions)
    {
        Solver = solver;
        OpNodeMemo = opNodeMemo;
        TileNodeMemo = tileNodeMemo;
        TileableNodeMemo = tileableNodeMemo;
        TargetOptions = targetOptions;
    }

    public Solver Solver { get; }

    public Dictionary<OpNode, OpNodeInfo> OpNodeMemo { get; }

    public Dictionary<TileNode, TileNodeInfo> TileNodeMemo { get; }

    public Dictionary<ITileAbleNode, DomainInfo> TileableNodeMemo { get; }

    public ITargetOptions TargetOptions { get; }
}
