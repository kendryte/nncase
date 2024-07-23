// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Google.OrTools.ConstraintSolver;

namespace Nncase.Schedule.TileTree;

public abstract class TreeSolverBase
{
    public TreeSolverBase(Solver solver, Dictionary<OpNode, OpNodeInfo> primitiveBufferInfo, Dictionary<TileNode, TileNodeInfo> levelBufferInfos, Dictionary<ITileAbleNode, DomainInfo> domainInfos, ITargetOptions targetOptions)
    {
        Solver = solver;
        OpNodeMemo = primitiveBufferInfo;
        TileNodeMemo = levelBufferInfos;
        TileableNodeMemo = domainInfos;
        TargetOptions = targetOptions;
    }

    public Solver Solver { get; }

    public Dictionary<OpNode, OpNodeInfo> OpNodeMemo { get; }

    public Dictionary<TileNode, TileNodeInfo> TileNodeMemo { get; }

    public Dictionary<ITileAbleNode, DomainInfo> TileableNodeMemo { get; }

    public ITargetOptions TargetOptions { get; }
}
