// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Reactive;
using Google.OrTools.ConstraintSolver;
using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.TIR;
using Nncase.TIR.Builders;

namespace Nncase.Schedule.TileGraph;

public record class NodeWithBuffer(TileNode Node, BufferIdentity Id)
{
}

public sealed class TreeSolveResult : TreeSolverBase<long>, ITreeNodeVisitor<TreeSolveResult.Context, Unit>
{
    private readonly Dictionary<ITileable, Dictionary<BufferIdentity, SubViewInfo>> _subViewMemo;

    public TreeSolveResult(BufferGraph primBufferGraph, long objectiveValue, Dictionary<int, Dictionary<NodeWithBuffer, long>> levelNodeBufferBoxs, Dictionary<int, Dictionary<NodeWithBuffer, Tuple<int, int>>> levelTreeBufferLifeness, Dictionary<OpNode, OpNodeInfo<long>> primitiveBufferInfo, Dictionary<TileNode, TileNodeInfo<long>> levelBufferInfos, Dictionary<ITileable, DomainInfo<long>> domainInfos, ITargetOptions targetOptions)
        : base(null!, primitiveBufferInfo, levelBufferInfos, domainInfos, targetOptions)
    {
        PrimBufferGraph = primBufferGraph;
        ObjectiveValue = objectiveValue;
        LevelBufferSizes = levelNodeBufferBoxs;
        LevelBufferLifeness = levelTreeBufferLifeness;
        LevelBufferOffsets = new();
        PrimBufferMemo = new();
        _subViewMemo = new();
    }

    public Dictionary<BufferIdentity, TIR.Buffer> PrimBufferMemo { get; }

    public BufferGraph PrimBufferGraph { get; }

    public long ObjectiveValue { get; }

    public Dictionary<int, Dictionary<NodeWithBuffer, long>> LevelBufferSizes { get; }

    public Dictionary<int, Dictionary<NodeWithBuffer, ulong>> LevelBufferOffsets { get; }

    public Dictionary<int, Dictionary<NodeWithBuffer, Tuple<int, int>>> LevelBufferLifeness { get; }

    public Unit Visit(TileNode value, Context context)
    {
        var (parentbuilder, partentOffsets) = context;

        var loopBuilders = new ISequentialBuilder<TIR.For>[value.DomainRelation.Map.Results.Length];
        var loopVars = new Var[value.DomainRelation.Map.Results.Length];

        var nodeMemo = TileNodeMemo[value];

        // from inner to outer
        for (int i = value.DomainRelation.Map.Results.Length - 1; i >= 0; i--)
        {
            long stop = nodeMemo.BackWardExtents[0][i];
            long tileSize = TileableNodeMemo[value].TileVars[i];
            loopBuilders[i] = T.Serial(out var loopVar, (0L, stop, stop / tileSize), $"d{i}_Op{value.OpId}_L{value.Level}");
            loopVars[i] = loopVar;
        }

        var initOffsets = Enumerable.Repeat<Expr>(0L, loopVars.Length).ToArray();
        foreach (var (k, v) in TileableNodeMemo[value].DimsMap)
        {
            initOffsets[k] += partentOffsets[v];
        }

        // forwardOffsets[0] means partentOffsets, forwardOffsets[i] means partentOffsets[0:i] + loop vars[0:i]
        var forwardOffsets = new Expr[loopVars.Length + 1][];
        for (int i = 0; i < loopVars.Length + 1; i++)
        {
            var offsets = forwardOffsets[i] = initOffsets.ToArray();

            for (int j = 0; j < i; j++)
            {
                offsets[j] += loopVars[j];
            }
        }

        // var domainLetBuilders = Enumerable.Range(0, value.DimNames.Length).Select(i => new List<ISequentialBuilder<Expr>>()).ToArray();
        var cntBuilder = parentbuilder;
        for (int i = 0; i < value.DomainRelation.Map.Results.Length; i++)
        {
            foreach (var (bid, bufferInfo) in nodeMemo.BufferInfoMap)
            {
                var place = bufferInfo.Places[i];
                for (int sl = 0; sl < place.Length; sl++)
                {
                    if (place[sl] == 1)
                    {
                        var viewInfo = GetParentSubViewInfo(sl + 1, value, bid, bufferInfo.Map, forwardOffsets[i], bufferInfo.Shapes[i]);
                        var subView = viewInfo.InnerAllocated ? IR.F.Buffer.AllocateBufferView(viewInfo.Buffer) : IR.F.Buffer.BufferSubview(viewInfo.Buffer, viewInfo.Offsets, new IR.Tuple(viewInfo.Shape.Select(x => (Expr)x).ToArray()));
                        Var subBufVar;

                        // the parent buffer is temp buffer.
                        var letBuilder = T.Let(out subBufVar, subView, $"{bid}_L{value.Level}");
                        cntBuilder.Body(letBuilder);
                        cntBuilder = letBuilder;

                        if (!_subViewMemo.TryGetValue(value, out var subViewMap))
                        {
                            subViewMap = new();
                            _subViewMemo.Add(value, subViewMap);
                        }

                        subViewMap[bid] = new(subBufVar, viewInfo.Offsets);
                    }
                }
            }

            cntBuilder.Body(loopBuilders[i]);
            cntBuilder = loopBuilders[i];
        }

        foreach (var child in value.Children)
        {
            var childBuilder = T.Sequential();
            child.Accept(this, new(childBuilder, forwardOffsets[^1]));
            loopBuilders[^1].Body(childBuilder);
        }

        return default;
    }

    public Unit Visit(OpNode value, Context context)
    {
        var (parentbuilder, partentOffsets) = context;

        var buffers = new Expr[value.BufferShapes.Length];
        for (int i = 0; i < value.BufferShapes.Length; i++)
        {
            var bid = new BufferIdentity(value.Wrapped, i);
            var viewInfo = GetParentSubViewInfo(value.Level, value, bid, value.DomainRelation.Map * OpNodeMemo[value].Maps[i], partentOffsets, OpNodeMemo[value].Shapes[i]);

            buffers[i] = IR.F.Buffer.BufferSubview(viewInfo.Buffer, viewInfo.Offsets, new IR.Tuple(viewInfo.Shape.Select(x => (Expr)x).ToArray()));
        }

        var bodyVarReplaces = new Dictionary<Expr, Expr>();
        for (int i = 0; i < value.Grid.BodyParameters.Length; i++)
        {
            bodyVarReplaces.Add(value.Grid.BodyParameters[i], buffers[i]);
        }

        var domain = new IR.Tuple(partentOffsets.Select(off => new IR.Tuple(off, 0L)).ToArray());
        bodyVarReplaces.Add(value.Grid.DomainParameter, domain);
        var nestBody = new ReplacingExprCloner(bodyVarReplaces).Clone(value.Grid.Body, default);
        parentbuilder.Body(nestBody);

        return default;
    }

    public void ScheduleBuffers()
    {
        foreach (var (level, nodeBufferSizes) in LevelBufferSizes)
        {
            var nodeBufferOffsets = LevelBufferOffsets[level] = new();
            var solver = new Solver("buffer scheduler");
            var xstarts = new List<IntVar>();
            var xsizes = new List<long>();
            var ystarts = new List<IntVar>();
            var ysizes = new List<long>();
            var validKeys = new List<NodeWithBuffer>();

            foreach (var (key, size) in nodeBufferSizes)
            {
                if (size > 0)
                {
                    xstarts.Add(solver.MakeIntConst(LevelBufferLifeness[level][key].Item1));
                    xsizes.Add(LevelBufferLifeness[level][key].Item2 - LevelBufferLifeness[level][key].Item1);
                    ystarts.Add(solver.MakeIntVar(0, TargetOptions.MemoryCapacities[level] - size));
                    ysizes.Add(size);
                    validKeys.Add(key);
                }
            }

            solver.Add(solver.MakeNonOverlappingBoxesConstraint(xstarts.ToArray(), ystarts.ToArray(), xsizes.ToArray(), ysizes.ToArray()));
            var collector = solver.MakeFirstSolutionCollector();
            foreach (var item in ystarts)
            {
                collector.Add(item);
            }

            var decisionBuilder = solver.MakeDefaultPhase(ystarts.ToArray());
            var monitors = new List<SearchMonitor>() { collector, solver.MakeSolutionsLimit(1), };
            if (Diagnostics.DumpScope.Current.IsEnabled(Diagnostics.DumpFlags.Tiling))
            {
                monitors.Add(solver.MakeSearchLog(10000));
            }

            var status = solver.Solve(decisionBuilder, monitors.ToArray());
            if (!status)
            {
                throw new InvalidOperationException("can't schedule buffers!");
            }

            var sol = collector.Solution(0);
            for (int i = 0; i < ystarts.Count; i++)
            {
                nodeBufferOffsets[validKeys[i]] = (ulong)sol.Value(ystarts[i]);
            }
        }
    }

    private TensorType GetBufferTensorType(Expr expr)
    {
        TensorType GetTensorType(IRType type) => type switch
        {
            TensorType t => t,
            DistributedType dt => Utilities.DistributedUtility.GetDividedTensorType(dt),
            _ => throw new NotSupportedException(),
        };

        return expr switch
        {
            IR.Buffers.BufferOf bufof => GetTensorType(bufof.Input.CheckedType),
            Call { Target: IR.Buffers.Uninitialized } c => GetTensorType(c.CheckedType),
            _ => throw new NotSupportedException(),
        };
    }

    /// <summary>
    /// declare the input/output buffer.
    /// </summary>
    private TIR.Buffer GetParentDeclareBuffer(int storeLevel, ITileable node, BufferIdentity bid)
    {
        var expr = bid.Node.Grid.Buffers[bid.Index];
        var tensorType = GetBufferTensorType(expr);
        if (!PrimBufferMemo.TryGetValue(bid, out var buffer))
        {
            buffer = T.AttachBuffer(None.Default, tensorType, MemoryLocation.Data, 1, out _, $"{bid}");

            PrimBufferMemo.Add(bid, buffer);
        }

        return buffer;
    }

    private bool TryGetParerntBuffer(ITreeNode node, BufferIdentity bid, out Expr parentBuffer, out IR.Tuple parentOffsets)
    {
        var cbid = bid;
        var parentNode = node.Parent;
        while (parentNode is TileNode parentTileNode && parentTileNode.OpId != -1)
        {
            var pbid = TileNodeMemo[parentTileNode].GetCacheBid(cbid);
            if (_subViewMemo.TryGetValue(parentTileNode, out var subViewMap) && subViewMap.TryGetValue(pbid, out var subViewInfo))
            {
                parentBuffer = subViewInfo.Buffer;
                parentOffsets = subViewInfo.Offsets;
                return true;
            }

            parentNode = parentTileNode.Parent;
            cbid = pbid;
        }

        parentBuffer = null!;
        parentOffsets = null!;
        return false;
    }

    private ParentSubViewInfo GetParentSubViewInfo(int storeLevel, ITreeNode node, BufferIdentity bid, AffineMap map, Expr[] forwardOffsets, long[] shapeExprs)
    {
        var offset = new IR.Tuple(map.Apply(forwardOffsets, Enumerable.Repeat<Expr>(0, forwardOffsets.Length).ToArray()).Select(i => i.Start).ToArray());
        var shape = shapeExprs.Select(s => (int)s).ToArray();
        bool innerAllocated = false;
        if (TryGetParerntBuffer(node, bid, out var parentBuffer, out var parentOffsets))
        {
            var subOffset = new Expr[offset.Count];
            for (int j = 0; j < subOffset.Length; j++)
            {
                var x = offset.Fields[j] - parentOffsets.Fields[j];
                subOffset[j] = x;

                // CompilerServices.ERewrite(x, new Passes.IRewriteRule[] { new Passes.Rules.Arithmetic.AssociateAdd(), new Passes.Rules.Arithmetic.CommutateAdd(), new Passes.Rules.Arithmetic.XNegX(), new Passes.Rules.Arithmetic.XNegX0() }, new(), CompileOptions);
            }

            offset = new IR.Tuple(subOffset);
        }
        else
        {
            var (outputs, inputs) = PrimBufferGraph.GetInputsOutputs();
            if (outputs.Contains(bid))
            {
                parentBuffer = GetParentDeclareBuffer(storeLevel, node, bid);
            }
            else if (inputs.Contains(bid))
            {
                parentBuffer = GetParentDeclareBuffer(storeLevel, node, bid);
            }
            else if (node is TileNode tileNode)
            {
                parentBuffer = GetParentAllocateBuffer(storeLevel, tileNode, bid, shape, out innerAllocated);
            }
        }

        return new ParentSubViewInfo(parentBuffer, offset, shape, innerAllocated);
    }

    /// <summary>
    /// Get the local allocate buffer.
    /// </summary>
    private TIR.Buffer GetParentAllocateBuffer(int storeLevel, TileNode node, BufferIdentity bid, int[] shape, out bool innerAllocated)
    {
        var expr = bid.Node.Grid.Buffers[bid.Index];
        var tensorType = GetBufferTensorType(expr);
        innerAllocated = false;
        if (!PrimBufferMemo.TryGetValue(bid, out var buffer))
        {
            TileNode rootNode = node;
            while (rootNode.Parent is TileNode parentTileNode && parentTileNode.OpId != -1)
            {
                rootNode = parentTileNode;
            }

            if (storeLevel < rootNode.Level)
            {
                tensorType = new TensorType(tensorType.DType, shape); // according to subtensor shape.
                var start = LevelBufferOffsets[storeLevel][new(node, bid)];
                buffer = T.AttachBuffer(Tensor.FromPointer(start, tensorType.DType), tensorType, MemoryLocation.L1Data, 1, out _, $"{bid}");
                innerAllocated = true;
            }
            else
            {
                buffer = T.AttachBuffer(None.Default, tensorType, MemoryLocation.Data, 1, out _, $"{bid}");
            }

            PrimBufferMemo.Add(bid, buffer);
        }

        return buffer;
    }

    public sealed record Context(ISequentialBuilder<Expr> ParentBuilder, Expr[] ForwardOffsets)
    {
    }

    public sealed record ParentSubViewInfo(Expr Buffer, IR.Tuple Offsets, int[] Shape, bool InnerAllocated)
    {
    }

    public sealed record SubViewInfo(Expr Buffer, IR.Tuple Offsets)
    {
    }
}
