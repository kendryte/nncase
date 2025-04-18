// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Reactive;
using Google.OrTools.Sat;
using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.TIR;
using Nncase.TIR.Builders;
using static Nncase.TIR.TIRExtensions;

namespace Nncase.Schedule.TileGraph;

public record class NodeWithBuffer(TileNode Node, BufferIdentity Id)
{
}

public sealed class TreeSolveResult : TreeSolverBase<long>, ITreeNodeVisitor<TreeSolveResult.Context, Unit>
{
    private readonly Dictionary<ITileable, Dictionary<BufferIdentity, SubViewInfo>> _subViewMemo;

    public TreeSolveResult(BufferGraph primBufferGraph, long objectiveValue, Dictionary<int, Dictionary<NodeWithBuffer, long>> levelNodeBufferBoxs, Dictionary<int, Dictionary<NodeWithBuffer, Tuple<int, int>>> levelTreeBufferLifeness, Dictionary<OpNode, OpNodeInfo<long>> primitiveBufferInfo, Dictionary<TileNode, TileNodeInfo<long>> levelBufferInfos, Dictionary<ITileable, DomainInfo<long>> domainInfos, ICpuTargetOptions targetOptions, string moduleKind)
        : base(null!, primitiveBufferInfo, levelBufferInfos, domainInfos, targetOptions)
    {
        PrimBufferGraph = primBufferGraph;
        (Inputs, Outputs) = primBufferGraph.GetInputsOutputs();
        ObjectiveValue = objectiveValue;
        LevelBufferSizes = levelNodeBufferBoxs;
        LevelBufferLifeness = levelTreeBufferLifeness;
        ModuleKind = moduleKind;
        LevelBufferOffsets = new();
        PrimBufferMemo = new();
        _subViewMemo = new();
    }

    public Dictionary<BufferIdentity, Expr> PrimBufferMemo { get; }

    public BufferGraph PrimBufferGraph { get; }

    public HashSet<BufferIdentity> Inputs { get; }

    public HashSet<BufferIdentity> Outputs { get; }

    public long ObjectiveValue { get; }

    public Dictionary<int, Dictionary<NodeWithBuffer, long>> LevelBufferSizes { get; }

    public Dictionary<int, Dictionary<NodeWithBuffer, ulong>> LevelBufferOffsets { get; }

    public Dictionary<int, Dictionary<NodeWithBuffer, Tuple<int, int>>> LevelBufferLifeness { get; }

    public string ModuleKind { get; }

    public Unit Visit(TileNode value, Context context)
    {
        var (parentbuilder, partentOffsets) = context;

        var loopBuilders = new ISequentialBuilder<TIR.For>[value.DomainRelation.Map.Results.Length];
        var loopVars = new DimVar[value.DomainRelation.Map.Results.Length];

        var nodeMemo = TileNodeMemo[value];

        // from inner to outer
        for (int i = value.DomainRelation.Map.Results.Length - 1; i >= 0; i--)
        {
            long stop = nodeMemo.BackWardExtents[0][i];
            long tileSize = TileableNodeMemo[value].TileVars[i];
            loopBuilders[i] = T.Serial(out var loopVar, (0L, stop, stop / tileSize), $"d{i}_Op{value.OpId}_L{value.Level}");
            loopVars[i] = loopVar;
        }

        var initOffsets = Enumerable.Repeat<Dimension>(0L, loopVars.Length).ToArray();
        foreach (var (k, v) in TileableNodeMemo[value].DimsMap)
        {
            initOffsets[k] += partentOffsets[v];
        }

        // forwardOffsets[0] means partentOffsets, forwardOffsets[i] means partentOffsets[0:i] + loop vars[0:i]
        var forwardOffsets = new Dimension[loopVars.Length + 1][];
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
        for (int i = 0; i < loopVars.Length + 1; i++)
        {
            foreach (var (bid, bufferInfo) in nodeMemo.BufferInfoMap)
            {
                var place = bufferInfo.Places[i];
                var expr = bid.Node.Grid.Buffers[bid.Index];
                var distributedType = GetBufferDistributedType(expr);
                for (int sl = 0; sl < place.Length; sl++)
                {
                    // skip the top level allocate
                    if (!(value.Level == PrimBufferGraph.Level && i == 0 && sl == (PrimBufferGraph.Level - 1)) && place[sl] == 1)
                    {
                        var kernelInfo = bid.Node.GetKernelInfo(TargetOptions);
                        var viewInfo = GetParentSubViewInfo(sl, value, bid, bufferInfo.Map, forwardOffsets[i], bufferInfo.Shapes[i]);
                        Expr subView;
                        if (viewInfo.InnerAllocated)
                        {
                            subView = IR.F.Buffer.AllocateBufferView(viewInfo.Buffer);
                        }
                        else
                        {
                            // for cpu we can use tensor view.
                            if (TargetOptions.UnifiedMemoryArch)
                            {
                                subView = IR.F.Buffer.BufferSubview(viewInfo.Buffer, viewInfo.Offsets, new IR.Tuple(viewInfo.Shape.Select(x => (Expr)x).ToArray()));
                            }
                            else
                            {
                                // for device we should use copy.
                                var offset = LevelBufferOffsets[sl][new(value, bid)];
                                var dtype = viewInfo.Buffer.CheckedDataType;
                                var shape = bufferInfo.Shapes[i].Select(i => (Dimension)i).ToArray();
                                subView = new TIR.Buffer($"{bid}_L{value.Level}_Copy", dtype, new MemSpan(Tensor.FromPointer(offset, dtype), bufferInfo.SizeVars[i], MemoryLocation.Data, 0), shape, TensorUtilities.GetStrides(shape), distributedType);
                            }
                        }

                        Var subViewVar;

                        // the parent buffer is temp buffer.
                        var letBuilder = T.Let(out subViewVar, subView, $"{bid}_L{value.Level}");
                        if (!TargetOptions.UnifiedMemoryArch)
                        {
                            var srcBufView = IR.F.Buffer.BufferSubview(viewInfo.Buffer, viewInfo.Offsets, new IR.Tuple(viewInfo.Shape.Select(x => (Expr)x).ToArray()));
                            if (kernelInfo.BufferInfos[bid.Index].State.HasFlag(MicroKernelBufferInfo.BufferState.Read))
                            {
                                if (bid.Node.Op.GetType().Name.Contains("Matmul", StringComparison.Ordinal) && bid.IsOutput)
                                {
                                    var kdim = bid.Node.WriteAccess.Domains.Length - 2;
                                    var val = value;
                                    bool isLoopRelated = false;
                                    while (val.Parent is TileNode parent)
                                    {
                                        var m = TileableNodeMemo[val];
                                        if (val.Level == value.Level)
                                        {
                                            if (i > kdim && m.TileVars[kdim] != 1)
                                            {
                                                isLoopRelated = true;
                                                break;
                                            }
                                        }
                                        else
                                        {
                                            if (m.TileVars[kdim] != 1)
                                            {
                                                isLoopRelated = true;
                                                break;
                                            }
                                        }

                                        val = parent;
                                    }

                                    if (isLoopRelated)
                                    {
                                        letBuilder.Body(T.Memcopy(subViewVar, srcBufView));
                                    }
                                }
                                else
                                {
                                    letBuilder.Body(T.Memcopy(subViewVar, srcBufView));
                                }
                            }

                            if (kernelInfo.BufferInfos[bid.Index].State.HasFlag(MicroKernelBufferInfo.BufferState.Write))
                            {
                                letBuilder.Tail(T.Memcopy(srcBufView, subViewVar));
                            }
                        }

                        cntBuilder.Body(letBuilder);
                        cntBuilder = letBuilder;

                        if (!_subViewMemo.TryGetValue(value, out var subViewMap))
                        {
                            subViewMap = new();
                            _subViewMemo.Add(value, subViewMap);
                        }

                        subViewMap[bid] = new(subViewVar, viewInfo.Offsets);
                    }
                }
            }

            if (i < loopVars.Length)
            {
                cntBuilder.Body(loopBuilders[i]);
                cntBuilder = loopBuilders[i];
            }
        }

        foreach (var child in value.Children)
        {
            var childBuilder = T.Sequential();
            child.Accept(this, new(childBuilder, forwardOffsets[^1]));
            cntBuilder.Body(childBuilder);
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
            var model = new CpModel();
            var rectangles = new Dictionary<NodeWithBuffer, (IntervalVar XInterval, IntervalVar YInterval)>();
            int count = 0;
            var cons = model.AddNoOverlap2D();
            foreach (var (key, size) in nodeBufferSizes)
            {
                if (size > 0)
                {
                    var x = model.NewFixedSizeIntervalVar(LevelBufferLifeness[level][key].Item1, LevelBufferLifeness[level][key].Item2 - LevelBufferLifeness[level][key].Item1, $"x{count}");
                    var ystart = model.NewIntVar(0, TargetOptions.MemoryCapacities[level] - size, $"ystart{count}");
                    if (ModuleKind == "xpu")
                    {
                        model.AddModuloEquality(0, ystart, 128);
                    }

                    var y = model.NewFixedSizeIntervalVar(ystart, size, $"y{count}");
                    cons.AddRectangle(x, y);
                    rectangles.Add(key, (x, y));
                    count++;
                }
            }

#if false
            // process inplace buffer.
            foreach (var (k, (x, y)) in rectangles)
            {
                var inplaceMemo = k.Id.Node.Op.GetInPlaceMemo();
                if (!inplaceMemo.TryGetValue(k.Id.Index, out var sourceIndex))
                {
                    continue;
                }

                // 1. when source buffer is isolated. we can find it in rectangles.
                foreach (var sourceKey in rectangles.Keys.Where(n => ReferenceEquals(n.Node, k.Node) && n.Id.Index == sourceIndex))
                {
                    model.Add(rectangles[sourceKey].YInterval.StartExpr() == y.StartExpr());
                }

                // 2. source buffer has been reused. we need find it in defuseMap firstly.
                foreach (var (defId, _) in TileNodeMemo[k.Node].DefUseMap.Where(kv => ReferenceEquals(kv.Value.Node, k.Id.Node) && kv.Value.Index == sourceIndex))
                {
                    foreach (var defKey in rectangles.Keys.Where(n => ReferenceEquals(n.Node, k.Node) && n.Id == defId))
                    {
                        model.Add(rectangles[defKey].YInterval.StartExpr() == y.StartExpr());
                    }
                }
            }
#endif

            var solver = new CpSolver();
            var status = solver.Solve(model);
            if (status is not CpSolverStatus.Optimal)
            {
                throw new InvalidOperationException("can't schedule buffers!");
            }

            foreach (var (k, (_, y)) in rectangles)
            {
                nodeBufferOffsets[k] = (ulong)solver.Value(y.StartExpr());
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

    private DistributedType GetBufferDistributedType(Expr expr)
    {
        DistributedType GetTensorType(IRType type) => type switch
        {
            TensorType => null!,
            DistributedType dt => dt,
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
    /// get declare of the input/output buffer which was stored on top level.
    /// </summary>
    private Expr GetTopLevelDeclareBuffer(BufferIdentity bid)
    {
        var expr = bid.Node.Grid.Buffers[bid.Index];
        var tensorType = GetBufferTensorType(expr);

        // TODO: Currently we only support the buffer which is not distributed.
        // var distributedType = GetBufferDistributedType(expr);
        if (!PrimBufferMemo.TryGetValue(bid, out var buffer))
        {
            buffer = new Var($"{bid}", tensorType);
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
            var pbid = TileNodeMemo[parentTileNode].GetByChildBuffer(cbid);
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

    private ParentSubViewInfo GetParentSubViewInfo(int storeLevel, ITreeNode node, BufferIdentity bid, AffineMap map, Dimension[] forwardOffsets, long[] shapeExprs)
    {
        var offset = new IR.Tuple(map.Apply(forwardOffsets, Enumerable.Repeat<Dimension>(0L, forwardOffsets.Length).ToArray()).Select(i => i.Start).ToArray());
        var shape = shapeExprs.ToArray();
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
                parentBuffer = GetTopLevelDeclareBuffer(bid);
            }
            else if (inputs.Contains(bid))
            {
                parentBuffer = GetTopLevelDeclareBuffer(bid);
            }
            else if (node is TileNode tileNode)
            {
                parentBuffer = GetInnerAllocateBuffer(storeLevel, tileNode, bid, shape, out innerAllocated);
            }
        }

        return new ParentSubViewInfo(parentBuffer, offset, shape, innerAllocated);
    }

    /// <summary>
    /// Allocate a buffer which store at inner level.
    /// </summary>
    private TIR.Buffer GetInnerAllocateBuffer(int storeLevel, TileNode node, BufferIdentity bid, long[] shape, out bool innerAllocated)
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

        return (TIR.Buffer)buffer;
    }

    public sealed record Context(ISequentialBuilder<Expr> ParentBuilder, Dimension[] ForwardOffsets)
    {
    }

    public sealed record ParentSubViewInfo(Expr Buffer, IR.Tuple Offsets, long[] Shape, bool InnerAllocated)
    {
    }

    public sealed record SubViewInfo(Expr Buffer, IR.Tuple Offsets)
    {
    }
}
