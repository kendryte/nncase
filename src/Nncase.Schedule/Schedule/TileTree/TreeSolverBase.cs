// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Google.OrTools.ConstraintSolver;

namespace Nncase.Schedule.TileTree;

public abstract class TreeSolverBase
{
    public TreeSolverBase(Solver solver, IntExpr one, IntExpr zero, IntExpr elem, Dictionary<OpNode, OpNodeInfo> primitiveBufferInfo, Dictionary<TileNode, TileNodeInfo> levelBufferInfos, Dictionary<ITileAbleNode, DomainInfo> domainInfos, ITargetOptions targetOptions)
    {
        Solver = solver;
        One = one;
        Zero = zero;
        Elem = elem;
        OpNodeMemo = primitiveBufferInfo;
        TileNodeMemo = levelBufferInfos;
        TileableNodeMemo = domainInfos;
        TargetOptions = targetOptions;
    }

    public Solver Solver { get; }

    public IntExpr One { get; }

    public IntExpr Zero { get; }

    public IntExpr Elem { get; }

    public Dictionary<OpNode, OpNodeInfo> OpNodeMemo { get; }

    public Dictionary<TileNode, TileNodeInfo> TileNodeMemo { get; }

    public Dictionary<ITileAbleNode, DomainInfo> TileableNodeMemo { get; }

    public ITargetOptions TargetOptions { get; }
}
