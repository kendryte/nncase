// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Reactive;
using Google.OrTools.Sat;
using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.TIR;
using Nncase.TIR.Builders;
using Nncase.Utilities;
using static Nncase.TIR.TIRExtensions;
using Isl = IntegerSetLibrary;

namespace Nncase.Schedule.TileGraph;

public record class NodeWithBuffer(TileNode Node, BufferIdentity Id)
{
}

public sealed class TreeSolveResult : TreeSolverBase<long>, ITreeNodeVisitor<TreeSolveResult.Context, Unit>
{
    private readonly Dictionary<ITileable, Dictionary<BufferIdentity, SubViewInfo>> _subViewMemo;

    public TreeSolveResult(BufferGraph primBufferGraph, long objectiveValue, Dictionary<int, Dictionary<NodeWithBuffer, long>> levelNodeBufferBoxs, Dictionary<int, Dictionary<NodeWithBuffer, Tuple<int, int>>> levelTreeBufferLifeness, Dictionary<OpNode, OpNodeInfo<long>> primitiveBufferInfo, Dictionary<TileNode, TileNodeInfo<long>> levelBufferInfos, Dictionary<ITileable, DomainInfo<long>> domainInfos, INTTTargetOptions targetOptions, string moduleKind)
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

    public RankedShape PartialShapeFromDomain(Isl.set parentDomain, Isl.set tiledDomain, Isl.map access, uint dim, Dictionary<string, Dimension> paramDimMap)
    {
        var domainRank = parentDomain.dim(Isl.dim_type.set);
        var shapeRank = access.dim(Isl.dim_type.out_);

        var parentMaxMpa = parentDomain.max_multi_pw_aff();
        var parentMinMpa = parentDomain.min_multi_pw_aff();
        var tiledMaxMpa = tiledDomain.max_multi_pw_aff();
        var tiledMinMpa = tiledDomain.min_multi_pw_aff();
        var accessMpa = new Isl.multi_pw_aff(access.as_pw_multi_aff());

        for (int i = (int)dim; i < domainRank; i++)
        {
            tiledMaxMpa = tiledMaxMpa.set_at(i, parentMaxMpa.at(i));
            tiledMinMpa = tiledMinMpa.set_at(i, parentMinMpa.at(i));
        }

        var bufferMaxMpa = accessMpa.pullback(tiledMaxMpa.add_constant(1));
        var bufferMinMpa = accessMpa.pullback(tiledMinMpa);
        var bufferShapeMpa = bufferMaxMpa.sub(bufferMinMpa);
        var dimensions = new Dimension[shapeRank];
        for (int i = 0; i < bufferShapeMpa.size(); i++)
        {
            if (accessMpa.at(i).is_cst())
            {
                dimensions[i] = 1;
            }
            else
            {
                var pa = bufferShapeMpa.at(i);
                dimensions[i] = ISLUtility.ToDimension(pa, paramDimMap);
            }
        }

        return new RankedShape(dimensions);
    }

    public Unit Visit(TileNode value, Context context)
    {
        var (parentbuilder, parentOffsets, parentExtents) = context;

        // get current tile node's domain.
        // todo use domain map to introduce the new dimensions.
        var parentDomain = ISLUtility.ToParametricDomain(parentExtents, out var paramVarMap);
        var paramDimMap = paramVarMap.Select(p => (p.Key.Name, p.Value)).Concat(parentExtents.Select((d, i) => ($"d{i}", d))).ToDictionary();

        var loopBuilders = new ISequentialBuilder<TIR.For>[value.DomainRelation.Map.Results.Length];
        var loopVars = new DimVar[value.DomainRelation.Map.Results.Length];

        var nodeMemo = TileNodeMemo[value];
        var domainRank = value.DomainRelation.Map.Results.Length;

        // create tile map from tile vars
        Isl.map tilemap;
        {
            var dims = new List<string>();
            var outerDims = new List<string>();
            var innerDims = new List<string>();
            var constraints = new List<string>();
            for (int i = 0; i < value.DomainRelation.Map.Results.Length; i++)
            {
                var tilesize = nodeMemo.BackWardExtents[0][i] / TileableNodeMemo[value].TileVars[i];
                dims.Add($"d{i}");
                outerDims.Add($"d{i}_out");
                innerDims.Add($"d{i}_in");
                constraints.Add($"d{i}_out = {tilesize} * (d{i} // {tilesize}) and d{i}_in = d{i} - d{i}_out");
            }

            tilemap = new Isl.map(Isl.ctx.Current, $"{{ [{string.Join(',', dims)}] -> [{string.Join(',', outerDims)},{string.Join(',', innerDims)}] : {string.Join(" and ", constraints)} }}");
        }

        var tiledParentDomain = tilemap.intersect_domain(parentDomain).range();
        var tiledChildDomain = tiledParentDomain.move_dims(Isl.dim_type.param, 0, Isl.dim_type.set, 0, (uint)domainRank);
        var childBoundsMpa = tiledChildDomain.max_multi_pw_aff().add_constant(1).sub(tiledChildDomain.min_multi_pw_aff());

        // from inner to outer
        var forwardExtents = new Dimension[domainRank];
        for (int i = value.DomainRelation.Map.Results.Length - 1; i >= 0; i--)
        {
            Dimension start = 0L;
            Dimension stop = parentExtents[i];
            Dimension stride = nodeMemo.BackWardExtents[0][i] / TileableNodeMemo[value].TileVars[i];
            loopBuilders[i] = T.Serial(out var loopVar, (0L, stop, stride), $"d{i}_Op{value.OpId}_L{value.Level}");
            loopVars[i] = loopVar;
            paramDimMap.Add($"d{i}_out", loopVar);
            {
                Dimension forwardExtent;
                var boundPa = childBoundsMpa.at(i);
                if (boundPa.is_cst())
                {
                    forwardExtent = boundPa.max_val().num_si();
                }
                else
                {
                    var build = new Isl.ast_build(Isl.ctx.Current);
                    var astExpr = build.expr_from(boundPa);
                    forwardExtent = ISLUtility.ToDimension(astExpr, paramDimMap);
                    forwardExtent.Metadata = new()
                    {
                        Range = new(boundPa.min_val().num_si(), boundPa.max_val().num_si()),
                    };
                }

                forwardExtents[i] = forwardExtent;
            }
        }

        var initOffsets = Enumerable.Repeat<Dimension>(0L, loopVars.Length).ToArray();
        foreach (var (k, v) in TileableNodeMemo[value].DimsMap)
        {
            initOffsets[k] += parentOffsets[v];
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
                var accessMap = AffineUtility.AsMap(bufferInfo.Map);
                var distributedType = GetBufferDistributedType(expr);
                for (int sl = 0; sl < place.Length; sl++)
                {
                    // skip the top level allocate
                    if (!(value.Level == PrimBufferGraph.Level && i == 0 && sl == (PrimBufferGraph.Level - 1)) && place[sl] == 1)
                    {
                        var kernelInfo = bid.Node.GetKernelInfo(TargetOptions);

                        // calculate the buffer shape.
                        var partialShape = PartialShapeFromDomain(parentDomain, tiledChildDomain, accessMap, (uint)i, paramDimMap);
                        var viewInfo = GetParentSubViewInfo(sl, value, bid, bufferInfo.Map, forwardOffsets[i], partialShape);
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
                                subView = IR.F.Buffer.BufferSubview(viewInfo.Buffer, viewInfo.Offsets, viewInfo.Shape);
                            }
                            else
                            {
                                // for device we should use copy.
                                var offset = LevelBufferOffsets[sl][new(value, bid)];
                                var dtype = viewInfo.Buffer.CheckedDataType;
                                var physicalBuffer = new PhysicalBuffer(dtype.SizeInBytes, offset, bufferInfo.SizeVars[i], MemoryLocation.Data, 0);
                                subView = new TIR.Buffer($"{bid}_L{value.Level}_Copy", dtype, new MemSpan(physicalBuffer), viewInfo.Shape.ToArray(), TensorUtilities.GetDefaultStrides(bufferInfo.Shapes[i].Select(i => (Dimension)i).ToArray()), distributedType);
                            }
                        }

                        Var subViewVar;

                        // the parent buffer is temp buffer.
                        var letBuilder = T.Let(out subViewVar, subView, $"{bid}_L{value.Level}");
                        if (!TargetOptions.UnifiedMemoryArch)
                        {
                            var srcBufView = IR.F.Buffer.BufferSubview(viewInfo.Buffer, viewInfo.Offsets, viewInfo.Shape);
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
            child.Accept(this, new(childBuilder, forwardOffsets[^1], forwardExtents));
            cntBuilder.Body(childBuilder);
        }

        return default;
    }

    public Unit Visit(OpNode value, Context context)
    {
        var (parentbuilder, partentOffsets, parentExtents) = context;
        var parentDomain = ISLUtility.ToParametricDomain(parentExtents, out var paramVarMap);
        var paramDimMap = paramVarMap.Select(p => (p.Key.Name, p.Value)).Concat(parentExtents.Select((d, i) => ($"d{i}", d))).ToDictionary();

        var buffers = new Expr[value.BufferShapes.Length];
        for (int i = 0; i < value.BufferShapes.Length; i++)
        {
            var bid = new BufferIdentity(value.Wrapped, i);
            var shape = PartialShapeFromDomain(parentDomain, parentDomain, AffineUtility.AsMap(value.Grid.AccessMaps[i]), (uint)parentDomain.dim(Isl.dim_type.set), paramDimMap);
            var viewInfo = GetParentSubViewInfo(value.Level, value, bid, value.DomainRelation.Map * OpNodeMemo[value].Maps[i], partentOffsets, shape);

            buffers[i] = IR.F.Buffer.BufferSubview(viewInfo.Buffer, viewInfo.Offsets, viewInfo.Shape);
        }

        var bodyVarReplaces = new Dictionary<BaseExpr, BaseExpr>();
        for (int i = 0; i < value.Grid.BodyParameters.Length; i++)
        {
            bodyVarReplaces.Add(value.Grid.BodyParameters[i], buffers[i]);
        }

        var domain = new IR.Tuple(partentOffsets.Select(off => new IR.Tuple(IR.F.Shapes.AsTensor(off), (Expr)0L)).ToArray());
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

    private bool TryGetParerntBuffer(ITreeNode node, BufferIdentity bid, out Expr parentBuffer, out Shape parentOffsets)
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

    private ParentSubViewInfo GetParentSubViewInfo(int storeLevel, ITreeNode node, BufferIdentity bid, AffineMap map, Dimension[] forwardOffsets, RankedShape shape)
    {
        var offset = map.Apply(forwardOffsets, Enumerable.Repeat<Dimension>(0L, forwardOffsets.Length).ToArray()).Select(i => i.Start).ToArray();
        bool innerAllocated = false;
        if (TryGetParerntBuffer(node, bid, out var parentBuffer, out var parentOffsets))
        {
            var subOffset = new Dimension[offset.Length];
            for (int j = 0; j < subOffset.Length; j++)
            {
                var x = offset[j] - parentOffsets[j];
                subOffset[j] = x;

                // CompilerServices.ERewrite(x, new Passes.IRewriteRule[] { new Passes.Rules.Arithmetic.AssociateAdd(), new Passes.Rules.Arithmetic.CommutateAdd(), new Passes.Rules.Arithmetic.XNegX(), new Passes.Rules.Arithmetic.XNegX0() }, new(), CompileOptions);
            }

            offset = ISLUtility.RoundTrip(subOffset);
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
    private TIR.Buffer GetInnerAllocateBuffer(int storeLevel, TileNode node, BufferIdentity bid, RankedShape shape, out bool innerAllocated)
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

    public sealed record Context(ISequentialBuilder<Expr> ParentBuilder, Dimension[] ForwardOffsets, Dimension[] ForwardExtents)
    {
    }

    public sealed record ParentSubViewInfo(Expr Buffer, Shape Offsets, RankedShape Shape, bool InnerAllocated)
    {
    }

    public sealed record SubViewInfo(Expr Buffer, Shape Offsets)
    {
    }
}
