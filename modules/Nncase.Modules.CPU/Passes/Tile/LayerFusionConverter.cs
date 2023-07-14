// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

#if false
using System.Reactive;
using System.Runtime.CompilerServices;
using NetFabric.Hyperlinq;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.Buffers;
using Nncase.IR.F;
using Nncase.IR.K510;
using Nncase.IR.Math;
using Nncase.Passes.BufferSchedule;
using Nncase.Passes.Mutators;
using Nncase.Runtime.K510;
using Nncase.Schedule;
using Nncase.TIR;
using Nncase.TIR.Builders;
using Nncase.TIR.K510;
using Nncase.TIR.K510.Builders;
using Nncase.TIR.K510.Instructions;
using Buffer = Nncase.TIR.Buffer;
using MathF = Nncase.IR.F.Math;
using Range = Nncase.TIR.Range;
using Tuple = Nncase.IR.Tuple;

namespace Nncase.Passes.Tile;

public sealed class BufferRegionView
{
    private Expr? _cache;

    private Expr[]? _condtion_buffer_regions;

    private Expr[]? _region_size;

    public BufferRegionView(IEnumerable<Buffer> buffers, IEnumerable<Range> bounds, IEnumerable<Range> region, IndexMapKey key)
        : this(buffers, bounds, region, key, 0, null)
    {
    }

    public BufferRegionView(IEnumerable<Buffer> buffers, IEnumerable<Range> bounds, IEnumerable<Range> region, IndexMapKey key, Expr loopCount, int? promote)
    {
        Buffers = buffers.ToArray();
        Region = region.ToArray();
        LoopCount = loopCount;
        Parent = null;
        Key = key;
        Promote = promote;
        Bounds = bounds.ToArray();
    }

    public IndexMapKey Key { get; }

    /// <summary>
    /// Gets 记录他的loop count.
    /// </summary>
    public Expr LoopCount { get; }

    public int? Promote { get; }

    public IReadOnlyList<Range> Bounds { get; }

    public IReadOnlyList<Buffer> Buffers { get; }

    public IReadOnlyList<Range> Region { get; }

    public BufferRegionView? Parent { get; set; }

    public ReadOnlySpan<Expr> Dimensions => Buffers[0].Dimensions;

    /// <summary>
    /// Gets 返回带有condition的buffer region的表达式.
    /// </summary>
    public IReadOnlyList<Expr> BufferRegions
    {
        get
        {
            _condtion_buffer_regions ??= Buffers.Count == 0 ? Array.Empty<Expr>() : Buffers.Select(b => new BufferRegion(b, Region.ToArray())).ToArray();
            return _condtion_buffer_regions;
        }
    }

    public BufferRegionView this[params Range[] ranges]
    {
        get => new(Buffers, Bounds, Region.Zip(ranges).Select(tp => tp.Second.Equals(Range.All) ? tp.First : tp.Second.Stop switch { Call { Target: Unary { UnaryOp: UnaryOp.Neg } } => throw new NotSupportedException("Neg Region!"), _ => tp.Second, }), Key, LoopCount, Promote) { Parent = Parent is null ? this : Parent, }; // if stop is neg, add the shape, else return the origin range.
    }

    /// <summary>
    /// convert the BufferRegionView to expr.
    /// <remarks>
    /// 当开启ping pong时,如果
    /// </remarks>
    /// </summary>
    /// <param name="view">view.</param>
    public static implicit operator Expr(BufferRegionView view)
    {
        if (view._cache is not null)
        {
            return view._cache;
        }

        Expr expr;
        if (view.Buffers.Count == 0)
        {
            expr = IR.None.Default;
        }
        else if (view.Buffers.Count == 1)
        {
            expr = view.BufferRegions[0];
        }
        else if (view.Buffers.Count >= 2)
        {
            expr = new Tuple(view.BufferRegions.ToArray())[view.LoopCount % view.Buffers.Count];
        }
        else
        {
            throw new NotSupportedException();
        }

        view._cache = expr;
        return view._cache;
    }

    public static BufferRegionView None(IndexMapKey key) => new(Array.Empty<Buffer>(), new IRArray<Range>(), new IRArray<Range>(), key);

    public ReadOnlySpan<Expr> RegionSize()
    {
        _region_size ??= Region.AsValueEnumerable().Select(r => r.Stop - r.Start).ToArray();
        return _region_size;
    }

    public Expr RegionSize(int i) => RegionSize()[i];
}

/// <summary>
/// name 分配器.
/// </summary>
internal sealed class NameAllocator
{
    public Dictionary<string, int> NamePool { get; } = new();

    public string Get(string name)
    {
        if (!NamePool.TryGetValue(name, out var count))
        {
            count = 0;
        }

        NamePool[name] = count + 1;
        return count == 0 ? name : $"{name}_{count}";
    }
}

/// <summary>
/// buffer region view.
/// </summary>
internal sealed class LogiclPrimFuncCloner : ExprCloner<Unit>
{
    protected override Expr VisitLeafLogicalBuffer(LogicalBuffer buffer, Unit context)
    {
        return buffer;
    }

    protected override Expr VisitLeafPhysicalBuffer(PhysicalBuffer buffer, Unit context)
    {
        return buffer;
    }

    protected override Expr VisitVar(Var var, Unit context)
    {
        return var;
    }
}

internal sealed record ReIndexCacheKey(IBoundsInferGraph BoundsInferGraph, IndexMapKey From, IndexMapKey To, IRArray<TIR.Range> FromRegion, int? Promote)
{
}

internal abstract class LayerFusionConverter
{
    public NameAllocator NameAllocator { get; } = new();

    /// <summary>
    /// Gets map the graph expression and it's bufferRegion.
    /// </summary>
    public Dictionary<IndexMapKey, BufferRegionView> KeyToViewMap { get; } = new();

    /// <summary>
    /// Gets because of the index map can't create by var, so need other map save the relationship.
    /// </summary>
    public Dictionary<Var, IndexMapKey> VarToKeyMap { get; } = new(ReferenceEqualityComparer.Instance);

    /// <summary>
    /// Gets tile size 的变量.
    /// </summary>
    public List<Var> TileSizeVars { get; } = new();

    /// <summary>
    /// Gets loop 变量.
    /// </summary>
    public List<Var> LoopVars { get; } = new();

    /// <summary>
    /// Gets loop domains.
    /// </summary>
    public List<Range> LoopDomains { get; } = new();

    /// <summary>
    /// Gets 所有的blocks
    /// 最终是:
    /// mainBlock
    ///   loop n
    ///     block n
    ///       loop c
    ///         block c
    ///          .
    ///          .
    /// </summary>
    public List<ITileBlockBuilder> NestedBlocks { get; } = new();

    /// <summary>
    /// Gets nested loops.
    /// </summary>
    public List<ISequentialBuilder<For>> NestedLoops { get; } = new();

    public TileOptions TileOptions { get; protected set; } = null!;

    /// <summary>
    /// Gets or sets 默认的bounds infer graph.
    /// </summary>
    public abstract IBoundsInferGraph BoundsInferGraph { get; protected set; }

    /// <summary>
    /// Gets or sets 总的loop count.
    /// </summary>/
    public abstract Expr LoopCount { get; protected set; }

    /// <summary>
    /// Gets or sets ping pong 外层的tiling.
    /// </summary>
    public abstract Expr LoopCountOuter { get; protected set; }

    /// <summary>
    /// Gets or sets ping pong 内侧的tiling.
    /// </summary>
    public abstract Expr LoopCountInner { get; protected set; }

    /// <summary>
    /// Gets or sets 当前的fusion.
    /// </summary>
    public abstract Fusion CurrentFusion { get; protected set; }

    /// <summary>
    /// Gets glb reindex cache.
    /// </summary>
    protected Dictionary<ReIndexCacheKey, (IRArray<TIR.Range> ToRegion, IReadOnlyList<(Expr Before, Expr After)> Paddings)> GlbReindexCache { get; } = new();

    public abstract Expr Visit(Fusion fusion);

    public virtual PrimFunction BuildLogicalPrimFunc(Expr bodySeq)
    {
        var inputs_buffer = CurrentFusion.Parameters.ToArray().Select(p => (PhysicalBuffer)KeyToViewMap[VarToKeyMap[p]].Buffers[0]);
        var primFuncBuilder = T.PrimFunc(CurrentFusion.Name, K510RTModule.Kind, inputs_buffer.Concat(new[] { (PhysicalBuffer)KeyToViewMap[(Call)CurrentFusion.Body].Buffers[0] }).ToArray());

        NestedBlocks[^1].Body(bodySeq);
        primFuncBuilder.Body(
          I.MmuConf(0, 0, MMU_CONF_WIDTH._8, 0, ExtCompilerServices.Env.GlbDepth), // 把整个glb当作连续内存使用.
          NestedBlocks[0],
          I.Fence());

        var logicalPrimFunc = primFuncBuilder.Build();
        logicalPrimFunc = (PrimFunction)new Mutators.SimplifyBounds().Rewrite(logicalPrimFunc);
        logicalPrimFunc.InferenceType();
        GlbReindexCache.Clear();
        return logicalPrimFunc;
    }

    public abstract bool BalanceTileSize(int[] tile_size, Segment[] search_spaces);

    public virtual PrimFunction BuildPhysicalPrimFunc(int[] final_tile_size, IReadOnlyDictionary<Buffer, PhysicalBuffer> sched_result, PrimFunction logicalPrimFunc)
    {
        var physicalizer = new BufferPhysicalizer(final_tile_size, sched_result, TileSizeVars);
        var physicalPrimFunc = (PrimFunction)physicalizer.Rewrite(logicalPrimFunc);
        return physicalPrimFunc;
    }

    public virtual int[] SearchTileSize(ISearchTileGenerator tile_generator, PrimFunction logicalPrimFunc, bool multi_workers, bool hasResult, out ScheduledResponse response)
    {
        AllocationCache<int[], ScheduledResponse> response_cache = new();
        bool schedule_status = false;
        int[] final_tile = Array.Empty<int>();

        while (true)
        {
            var next_tile = tile_generator.GetNextTile(schedule_status).ToArray();
            if (next_tile.Length == 0)
            {
                break;
            }

            schedule_status = TryScheduleNextTileSize(next_tile, logicalPrimFunc, response_cache, multi_workers, hasResult);
            if (schedule_status)
            {
                final_tile = next_tile;
                response_cache.CheckIn();
            }
        }

        if (!final_tile.Any())
        {
            response = new(new Dictionary<TIR.Buffer, TIR.PhysicalBuffer>(), null!, null!, logicalPrimFunc, null!, 0, false);
            return final_tile;
        }

        // take back last success allocation result
        response = response_cache.GetLastSuccess(final_tile);
        return final_tile;
    }

    public virtual Expr Visit(IndexMapKey mapKey, string prefix, int? promote = null)
    {
        prefix = prefix + mapKey.Prefix;
        return mapKey.Expr switch
        {
            Call call => (call.Target switch
            {
                GNNELoad op => LowerGnneLoad(mapKey, call, op, NameAllocator.Get(nameof(GNNELoad)), prefix, promote, true),
                GNNEStore op => LowerGnneStore(call, op, NameAllocator.Get(nameof(GNNEStore)), prefix),
                GNNEConv2D op => LowerGnneConv2D(mapKey, call, op, NameAllocator.Get(nameof(GNNEConv2D)), prefix),
                GNNEConv2DTranspose op => LowerGnneConv2DTranspose(mapKey, call, op, NameAllocator.Get(nameof(GNNEConv2D)), prefix),
                GNNEReduce op => LowerGnneReduce(mapKey, call, op, NameAllocator.Get(nameof(GNNEReduce)), prefix),
                GNNEMeshNet op => LowerGnneMeshNet(mapKey, call, op, NameAllocator.Get(nameof(GNNEMeshNet)), prefix),
                GNNETranspose op => LowerGnneTranspose(mapKey, call, op, NameAllocator.Get(nameof(GNNETranspose)), prefix),
                GNNEActivation op => LowerGnneActivation(mapKey, call, op, NameAllocator.Get(nameof(GNNEActivation)), prefix),
                GNNEPdpReduce op => LowerGnnePdpReduce(mapKey, call, op, NameAllocator.Get(nameof(GNNEPdpReduce)), prefix),
                GNNECrop op => LowerGnneCrop(mapKey, call, op, NameAllocator.Get(nameof(GNNECrop)), prefix),
                Uninitialized => T.Sequential(),
                _ => throw new NotSupportedException(),
            }).Build(),
            _ => T.Nop(),
        };
    }

    /// <summary>
    /// 子偏移输入到bounds infer后反推子偏移.
    /// </summary>
    /// <param name="from">from.</param>
    /// <param name="to">to.</param>
    /// <param name="sub_paddings">sub_paddings.</param>
    /// <param name="partialFuncs">the partial compute funcs.</param>
    /// <returns>.</returns>
    public virtual BufferRegionView GlbReIndex(BufferRegionView from, BufferRegionView to, out IReadOnlyList<(Expr Before, Expr After)> sub_paddings, params (int Axis, Func<TIR.Range, TIR.Range> CallBack)[] partialFuncs)
    {
        var key = new ReIndexCacheKey(BoundsInferGraph, from.Key, to.Key, new(from.Region), to.Promote);
        IReadOnlyList<Range> to_region;
        if (partialFuncs.Length == 0 && GlbReindexCache.TryGetValue(key, out var result))
        {
            to_region = result.ToRegion;
            sub_paddings = result.Paddings;
        }
        else
        {
            to_region = TileUtilities.GetRelativeNoPadBounds(BoundsInferGraph, from.Key, to.Key, from.Region, to.Promote, partialFuncs, out sub_paddings);
            if (partialFuncs.Length == 0)
            {
                GlbReindexCache.Add(key, (new(to_region), sub_paddings));
            }
        }

        return to[to_region.ToArray()];
    }

    protected virtual bool TryScheduleNextTileSize(int[] next_tile_size, PrimFunction logicalPrimFunc, AllocationCache<int[], ScheduledResponse> response_cache, bool multi_workers, bool hasResult)
    {
        // 1. make one tile feed dict
        var feed_dict = next_tile_size.Select((s, i) =>
                    new[] { (LoopVars[i], (IValue)Value.FromTensor(Tensor.FromScalar(0))),
                     (TileSizeVars[i], (IValue)Value.FromTensor(Tensor.FromScalar(s))), }).
                    SelectMany(i => i).
                    ToDictionary(kv => kv.Item1, kv => kv.Item2);
        var sched_candidate = new Dictionary<Buffer, PhysicalBuffer>(ReferenceEqualityComparer.Instance);

        // 2. folding the tileblock op to the block
        PrimFunction new_logical_primfunc;
        using (var dumpScope = new DumpScope(NullDumpper.Instance))
        {
            var pass = new PrimFuncPass { Name = "FoldingTileBlock" };
            pass.Add<RemoveFoldOfMakers>(feed_dict);
            pass.Add<FoldCondition>();
            pass.Add<FoldTileBlock>();
            var task = pass.RunAsync(new LogiclPrimFuncCloner().Clone(logicalPrimFunc, default), new());
            task.Wait();
            new_logical_primfunc = task.Result;
        }

        BufferScheduler bufferScheduler = new(new_logical_primfunc);

        // 3. clloction buffers
        bufferScheduler.LifeTimeAnalysis();

        // compute the size in bytes
        foreach (var buffer in bufferScheduler.RecordBuffers)
        {
            var dimensions = buffer.Dimensions.ToArray().Select(d => d.Evaluate(feed_dict).AsTensor().ToScalar<int>()).ToArray();
            var strides = TensorUtilities.GetStrides(dimensions);
            var glb_strides = strides.Select(s => s * buffer.ElemType.SizeInBytes).ToArray();

            if (bufferScheduler.InnerConstraints[buffer] == ConstraintsMode.Channel &&

               // 当load psum的时候,如果shape过小, 那么不额外添加stride.
               !(buffer.Name.Split(".").Last().StartsWith(GNNEConv2D.PSum.Name) &&
                 dimensions[2] * dimensions[3] < 14 * 14))
            {
                glb_strides = TileUtilities.PaddingAvoidConflict(dimensions, glb_strides, 1);
                strides = glb_strides.Select(s =>
                {
                    if (s % buffer.ElemType.SizeInBytes != 0)
                    {
                        throw new NotSupportedException();
                    }

                    return s / buffer.ElemType.SizeInBytes;
                }).ToArray();
            }

            var size_n_byte = dimensions[0] * glb_strides[0];

            // todo 可以不用align到一整行, 到一个bank即可.
            var glb_size = TileUtilities.AlignBy(size_n_byte, ExtCompilerServices.Env.GlbBankWidth * ExtCompilerServices.Env.GlbWidth);
            var physical_candidate = new PhysicalBuffer(buffer.Name, buffer.ElemType, buffer.MemLocation, dimensions, strides, start: 0, size: glb_size);
            sched_candidate.Add(buffer, physical_candidate);
        }

        var respose = bufferScheduler.Schedule(sched_candidate, multi_workers, hasResult);
        response_cache.Add(next_tile_size, respose);
        return respose.Success;
    }

    /// <summary>
    /// 申请 buffer.
    /// NOTE 会自动添加到buffer map, 同时会记录他ddr 上的padding到字典中.
    /// 如果给定 ddr buf region, 那么默认glb buffer region则是通过ddr buffer load 进来的,此时glb buffer的region是减去过padding的.
    /// 如果promote到对应的循环后,那么申请buffer的时候在promote内部的循环都应该被调整到最大值.
    /// </summary>
    /// <param name="mapKey">mapKey.</param>
    /// <param name="region">region.</param>
    /// <param name="ping_pong">开启ping pong就会多开一块相同的buffer.</param>
    /// <param name="promote"> 如果promote为int,那么就会提升buffer到指定循环, 为-1那么就是整个计算块, 会忽略ping pong. </param>
    /// <param name="specificLoopBounds">specificLoopBounds.</param>
    /// <param name="name">name.</param>
    /// <returns>.</returns>
    /// <exception cref="NotSupportedException">NotSupportedException.</exception>
    /// <exception cref="System.ArgumentOutOfRangeException">System.ArgumentOutOfRangeException.</exception>
    protected virtual Expr GetBufferRegion(IndexMapKey mapKey, out BufferRegionView region, bool ping_pong = false, int? promote = null, Dictionary<IndexMapKey, List<Range>>? specificLoopBounds = null, [CallerArgumentExpression("region")] string name = "region")
    {
        if (name.StartsWith("var "))
        {
            name = name[4..];
        }

        if (KeyToViewMap.ContainsKey(mapKey))
        {
            region = KeyToViewMap[mapKey];
            return T.Nop();
        }

        name = NameAllocator.Get(name);
        switch (mapKey.Expr)
        {
            case TensorConst con:
                {
                    IEnumerable<Range> bounds;
                    IEnumerable<Range> clampedBounds;
                    if (promote is int promoteInt)
                    {
                        if (promoteInt == -1)
                        {
                            clampedBounds = mapKey.Expr.CheckedShape.Select(s => new Range(0, s.FixedValue, 1));
                            bounds = BoundsInferGraph[mapKey].Bounds;
                        }
                        else
                        {
                            if (specificLoopBounds is null || !specificLoopBounds.TryGetValue(mapKey, out var newBounds))
                            {
                                newBounds = K510TIRExtensions.PromotedBounds(promoteInt, BoundsInferGraph, mapKey, LoopVars, LoopDomains).ToList();
                            }

                            bounds = newBounds;
                            clampedBounds = TIRUtilities.ClampBounds(newBounds, mapKey.Expr.CheckedShape);
                        }
                    }
                    else
                    {
                        bounds = BoundsInferGraph[mapKey].Bounds;
                        clampedBounds = BoundsInferGraph[mapKey].ClampedBounds;
                    }

                    T.ConstBuffer(con, out var ddr_buffer, name);
                    if (ping_pong)
                    {
                        throw new NotSupportedException();
                    }

                    region = new BufferRegionView(new[] { ddr_buffer }, bounds, clampedBounds, mapKey);
                    break;
                }

            case Call call:
                {
                    // 1. 对于glb buffer来说, 他的总大小要跟着申请buffer维度来变化.
                    // note 实际上对于
                    List<Range> bounds;
                    Expr loopCount;
                    if (promote is int promoteInt)
                    {
                        if (promoteInt == -1)
                        {
                            bounds = mapKey.Expr.CheckedShape.Select(s => new Range(0, s.FixedValue, 1)).ToList();
                            loopCount = 0;
                        }
                        else
                        {
                            if (specificLoopBounds is null || !specificLoopBounds.TryGetValue(mapKey, out var newBounds))
                            {
                                newBounds = K510TIRExtensions.PromotedBounds(promoteInt, BoundsInferGraph, mapKey, LoopVars, LoopDomains).ToList();
                            }

                            bounds = newBounds;
                            loopCount = K510TIRExtensions.PromotedLoopCount(promoteInt, LoopVars, LoopDomains);
                        }
                    }
                    else
                    {
                        bounds = BoundsInferGraph[mapKey].Bounds.ToList();
                        loopCount = LoopCount;
                    }

                    // note 这里的bounds实际上会因为输入不同的var而被改变, 所以后面要获取dimension的地方需要注意.
                    var dimensions = bounds.Select(r => r.Stop - r.Start).Select((b, i) => MathF.Min(b, call.CheckedShape[i].FixedValue)).ToArray();
                    List<Buffer> glb_buffers = new();
                    if (ping_pong)
                    {
                        for (int i = 0; i < TileOptions.PingPongNum; i++)
                        {
                            glb_buffers.Add(new LogicalBuffer(name + $"(p{i})", call.CheckedDataType, MemoryLocation.L2Data, dimensions));
                        }
                    }
                    else
                    {
                        glb_buffers.Add(new LogicalBuffer(name, call.CheckedDataType, MemoryLocation.L2Data, dimensions));
                    }

                    // 对于glb_buffer来说, 默认region 从0 开始, 但是要减去输入ddr index 的padding.
                    var noPadBounds = TIRUtilities.ComputeNoPadBounds(bounds, TIRUtilities.ComputePaddings(bounds, mapKey.Expr.CheckedShape));
                    region = new BufferRegionView(glb_buffers, bounds, noPadBounds, mapKey, loopCount, promote);
                    break;
                }

            case Var v:
                {
                    // the different mapkey will point to same var: add(v,conv(v))
                    if (!VarToKeyMap.TryGetValue(v, out var old_map_key))
                    {
                        T.PhysicalBuffer(v.CheckedDataType, MemoryLocation.Input, v.CheckedShape.ToValueArray(), out var ddr_buffer, name);
                        IEnumerable<Range> clampedBounds;
                        IReadOnlyList<Range> bounds;
                        if (promote is int promoteInt)
                        {
                            if (promoteInt != -1)
                            {
                                if (specificLoopBounds is null || !specificLoopBounds.TryGetValue(mapKey, out var newBounds))
                                {
                                    newBounds = K510TIRExtensions.PromotedBounds(promoteInt, BoundsInferGraph, mapKey, LoopVars, LoopDomains).ToList();
                                }

                                bounds = newBounds;
                            }
                            else
                            {
                                bounds = mapKey.Expr.CheckedShape.Select(s => new Range(0, s.FixedValue, 1)).ToList();
                            }

                            clampedBounds = TIRUtilities.ClampBounds(bounds, mapKey.Expr.CheckedShape);
                        }
                        else
                        {
                            bounds = BoundsInferGraph[mapKey].Bounds;
                            clampedBounds = BoundsInferGraph[mapKey].ClampedBounds;
                        }

                        if (ping_pong)
                        {
                            throw new NotSupportedException();
                        }

                        region = new BufferRegionView(new[] { ddr_buffer }, bounds, clampedBounds, mapKey);
                        VarToKeyMap.Add(v, mapKey);
                    }
                    else
                    {
                        region = KeyToViewMap[old_map_key];
                    }

                    break;
                }

            case None none:
                region = BufferRegionView.None(mapKey);
                break;
            default:
                throw new NotSupportedException();
        }

        KeyToViewMap.Add(mapKey, region);
        return T.Nop();
    }

    /// <summary>
    /// promote 的逻辑, 根据值选择移动当前的buffer开在哪个循环.
    /// -1 表示在所有循环之外
    /// 0 表示在N循环内
    /// 3 表示在W循环内.
    ///
    /// </summary>
    /// <param name="parentKey">上一级传入的key.</param>
    /// <param name="call">call.</param>
    /// <param name="op">op.</param>
    /// <param name="block_name">block_name.</param>
    /// <param name="prefix">prefix.</param>
    /// <param name="promote">promote.</param>
    /// <param name="softPipeLine">is enable soft pipe line.</param>
    /// <returns>.</returns>
    protected virtual ISequentialBuilder<Sequential> LowerGnneLoad(IndexMapKey parentKey, Call call, GNNELoad op, string block_name, string prefix, int? promote, bool softPipeLine)
    {
        var call_input = IndexMapKey.Create(call, GNNELoad.Input);
        var call_deq = IndexMapKey.Create(call, GNNELoad.DeqParams);

        var seq = T.Sequential().Body(
          Visit(call_deq, prefix, promote),
          GetBufferRegion(call_input, out var ddr_ld_input, name: prefix + "." + TileNames.DdrInput, promote: promote), // loadif 的输入可能来自于const或输入
          GetBufferRegion(call_deq, out var glb_ld_qarg_input), // 只有promote到n循环外时,才不进行ping pong.
          GetBufferRegion(parentKey, out var glb_ld_output, promote == -1 ? false : TileOptions.PingPong, name: prefix, promote: promote)); // load if 要用的glb buffer

        var block = EAction.TileBlock(block_name).
          Alloc(promote is null ? glb_ld_output.Buffers : None.Default).
          Reads(ddr_ld_input.BufferRegions, glb_ld_qarg_input.BufferRegions).
          Writes(glb_ld_output.BufferRegions).
          Predicate(true).// todo 这里先不做局部加载, 后面再实现
          Body(// promote这里load用的是full region, 但是在字典中存的还应该是partial的, 因为后面是每个glb的tile在使用.
            softPipeLine ?
              (TileOptions.PingPong & (promote != -1) ? K510.PingPongSlot(block_name, glb_ld_output.LoopCount / TileOptions.PingPongNum, glb_ld_output.LoopCount % TileOptions.PingPongNum) : T.Nop()) :
              T.Nop(),
            EAction.LoadT(ddr_ld_input, glb_ld_output, glb_ld_qarg_input, op.DeqAxis));

        if (promote is int promoteIndex)
        {
            // 如果promote, 那么在这个循环的所有block外执行
            NestedBlocks[promoteIndex + 1].Init(block);
            NestedBlocks[promoteIndex + 1].Alloc(glb_ld_output.Buffers);
        }
        else
        {
            seq.Body(block);
        }

        return seq;
    }

    protected virtual ISequentialBuilder<Sequential> LowerGnneMeshNet(IndexMapKey parentKey, Call call, GNNEMeshNet target, string block_name, string prefix)
    {
        prefix = NameAllocator.Get(nameof(GNNEMeshNet));
        var call_in_a = IndexMapKey.Create(call, GNNEMeshNet.InputA);
        var call_in_b = IndexMapKey.Create(call, GNNEMeshNet.InputB);
        var call_in_seg0 = IndexMapKey.Create(call, GNNEMeshNet.SegFittingParam0);
        var call_in_seg1 = IndexMapKey.Create(call, GNNEMeshNet.SegFittingParam1);
        var seq = T.Sequential().Body(
          Visit(call_in_a, prefix),
          Visit(call_in_b, prefix),
          GetBufferRegion(call_in_a, out var meshnet_input_a),
          GetBufferRegion(call_in_b, out var meshnet_input_b),
          GetBufferRegion(call_in_seg0, out var meshnet_input_seg0),
          GetBufferRegion(call_in_seg1, out var meshnet_input_seg1),
          GetBufferRegion(parentKey, out var meshnet_output, TileOptions.PingPong, name: prefix),
          EAction.TileBlock(block_name).
            Alloc(meshnet_output.Buffers).
            Reads(
                meshnet_input_a.BufferRegions,
                meshnet_input_b.BufferRegions,
                meshnet_input_seg0.BufferRegions,
                meshnet_input_seg1.BufferRegions).
            Body(
              EAction.MeshNetCompute(
                  (Fusion)call[GNNEMeshNet.MeshFunc],
                  meshnet_input_a,
                  meshnet_input_b,
                  meshnet_input_seg0,
                  meshnet_input_seg1,
                  meshnet_output)));

        if (!(call[GNNEMeshNet.InputB] is None && call[GNNEMeshNet.NewShape] is None && call[GNNEMeshNet.SegFittingParam0] is None && call[GNNEMeshNet.SegFittingParam1] is None && !TileUtilities.MeshFuncHasConstants((Fusion)call[GNNEMeshNet.MeshFunc])))
        {
            foreach (var item in meshnet_output.Buffers)
            {
                item.Metadata = new TileMetadata() { StrideByShape = true };
            }
        }

        return seq;
    }

    protected virtual ISequentialBuilder<Sequential> LowerGnneStore(Call call, GNNEStore op, string block_name, string prefix, bool promoteQarg = true)
    {
        prefix = NameAllocator.Get(nameof(GNNEStore));
        var cropPadding = ((TensorConst)call[GNNEStore.CropPadding]).Value.Cast<int>();
        var channel = call.CheckedShape[1].FixedValue;
        bool is_quant_by_channel = false;
        if (call[GNNEStore.QuantParams] is Call { Target: GNNELoad } l_qarg && l_qarg[GNNELoad.Input] is TensorConst qarg)
        {
            _ = qarg.Value.Cast<QuantizeParam>();
            if (qarg[0] != qarg[channel - 1])
            {
                is_quant_by_channel = true;
            }
        }

        var outputShape = call.CheckedShape.ToValueArray();
        T.PhysicalBuffer(call.CheckedDataType, MemoryLocation.Output, outputShape, out var ddr_st_buffer, name: prefix + ".ddr_buffer");
        var bounds = BoundsInferGraph[call].Bounds;

        var (paddingHBefore, paddingHafter) = TileUtilities.ComputePadding(bounds[2] - cropPadding[0, 0], outputShape[2]);
        var (paddingWBefore, paddingWafter) = TileUtilities.ComputePadding(bounds[3] - cropPadding[1, 0], outputShape[3]);

        var newBounds = bounds.ToArray();
        newBounds[2] = newBounds[2] - cropPadding[0, 0];
        newBounds[3] = newBounds[3] - cropPadding[1, 0];
        var ddrRegion = TIRUtilities.ClampBounds(newBounds, outputShape);
        var ddr_st_output = new BufferRegionView(new[] { ddr_st_buffer }, BoundsInferGraph[call].Bounds, ddrRegion, call);
        KeyToViewMap.Add(call, ddr_st_output);

        var call_in = IndexMapKey.Create(call, GNNEStore.Input);
        var call_qarg = IndexMapKey.Create(call, GNNEStore.QuantParams);
        return T.Sequential().Body(
          Visit(call_in, prefix),
          Visit(call_qarg, prefix, promoteQarg ? -1 : null), // 多层是只按h切, 此时oc满的,默认promote.
          GetBufferRegion(call_in, out var glb_st_input),
          GetBufferRegion(call_qarg, out var glb_st_qarg_input),
          EAction.TileBlock(block_name).Reads(glb_st_input.BufferRegions, glb_st_qarg_input.BufferRegions).Body(
            EAction.StoreT(ddr_st_output, glb_st_input[.., .., (glb_st_input.Region[2].Start + paddingHBefore, glb_st_input.Region[2].Stop - paddingHafter), (glb_st_input.Region[3].Start + paddingWBefore, glb_st_input.Region[3].Stop - paddingWafter)], glb_st_qarg_input, null, is_quant_by_channel)));
    }

    protected virtual ISequentialBuilder<Sequential> LowerGnneReduce(IndexMapKey parentKey, Call call, GNNEReduce op, string block_name, string prefix)
    {
        prefix = NameAllocator.Get(nameof(GNNEReduce));
        var reduce_in = IndexMapKey.Create(call, GNNEReduce.Input);
        var seq = T.Sequential().Body(
          Visit(reduce_in, prefix),
          GetBufferRegion(reduce_in, out var gnne_reduce_input),
          GetBufferRegion(parentKey, out var gnne_reduce_output, TileOptions.PingPong, name: prefix),
          EAction.TileBlock(block_name).
            Alloc(gnne_reduce_output.Buffers).
            Reads(gnne_reduce_input.BufferRegions).Body(
              EAction.Reduce(
                gnne_reduce_input,
                gnne_reduce_output,
                call[GNNEReduce.InitValue],
                op.ReduceOp,
                op.ReduceDim)));

        return seq;
    }

    /// <summary>
    /// 对于dw卷积来说,每个ic对应一个oc, 因此让每个tcu计算一半的if.
    /// </summary>
    protected ITileBlockBuilder GNNEConv2DSharedNone(Call call, string block_name, BufferRegionView glb_w, BufferRegionView glb_if, BufferRegionView glb_act, BufferRegionView glb_psum, BufferRegionView glb_of, bool is_init_psum, string prefix, int? promote = null)
    {
        var init_psums = GetInitPSumBufferRegion(call, IndexMapKey.Create(call, GNNEConv2D.PSum), glb_psum, promote, prefix, 1, ExtCompilerServices.Env.TcuActNum, out var part_condition);

        var block = EAction.TileBlock(block_name).Reads(glb_w.BufferRegions, is_init_psum ? init_psums[0].BufferRegions.Concat(init_psums[1].BufferRegions.Select(b => MathF.Condition(EAction.FoldOfMarker(part_condition > 1), b))).ToArray() : glb_psum.BufferRegions, glb_if.BufferRegions, glb_act.BufferRegions).Writes(glb_of.BufferRegions).Body(
          T.Unrolled(out var kh, new(0, glb_w.Dimensions[2], ExtCompilerServices.Env.PuHeight)).Body(
            T.Unrolled(out var kw, new(0, glb_w.Dimensions[3], ExtCompilerServices.Env.PuKernelSpad)).Body(
              T.Let(out var m_once, 1).Body(
              T.Let(out var c_once, MathF.Select(MathF.Equal(m_once, glb_w.RegionSize(0)), MathF.Min(MathF.Min(ExtCompilerServices.Env.PuWidth / m_once, ExtCompilerServices.Env.PuHeight / MathF.Min(glb_w.Dimensions[2], ExtCompilerServices.Env.PuHeight)), glb_w.RegionSize(0)), 1)).Body(// note 我这里没有实现dw卷积的多输出channel的, 默认都是 1 ic : 1 oc.
                T.Let(out var tcu_oc_chunk, TileUtilities.Split(glb_of.RegionSize(1), ExtCompilerServices.Env.TcuActNum)).Body(// 1. determine tcu act num
                T.Let(out var n_active_tcu, TileUtilities.SplitTimes(glb_of.RegionSize(1), tcu_oc_chunk)).Body(// 2. broadcast action
                  EAction.TcuDmBroadCast(TcuDivideStrategy.NoShare),
                  T.Unrolled(out var tcu_oc, new(glb_of.Region[1].Start, glb_of.Region[1].Stop, tcu_oc_chunk)).Body(// 3. loop over tcus and config each tcu
                    T.Let(out var m_once_tcu, 1).Body(
                    T.Let(out var c_once_tcu, MathF.Min(ExtCompilerServices.Env.PuHeight / glb_w.RegionSize(2), MathF.Min(tcu_oc + tcu_oc_chunk, glb_of.Region[1].Stop) - tcu_oc)).Body(
                      EAction.TcuPuConfAct(
                        TileUtilities.GetTcuIndexBits(tcu_oc / tcu_oc_chunk),
                        GlbReIndex(glb_of[.., (tcu_oc, MathF.Min(tcu_oc + tcu_oc_chunk, glb_of.Region[1].Stop)), .., ..], glb_act, out _),
                        call[GNNEConv2D.FusedClamp][0],
                        call[GNNEConv2D.FusedClamp][1]),
                      EAction.TcuPuConf(
                        TileUtilities.GetTcuIndexBits(tcu_oc / tcu_oc_chunk),
                        GlbReIndex(glb_of[.., (tcu_oc, MathF.Min(tcu_oc + tcu_oc_chunk, glb_of.Region[1].Stop)), .., ..], glb_if, out _),
                        glb_of[.., (tcu_oc, MathF.Min(tcu_oc + tcu_oc_chunk, glb_of.Region[1].Stop)), .., ..],
                        IR.F.Math.Min(glb_w.Dimensions[2], ExtCompilerServices.Env.PuHeight),
                        IR.F.Math.Min(glb_w.Dimensions[3], ExtCompilerServices.Env.PuKernelSpad),
                        m_once: m_once_tcu,
                        c_once: c_once_tcu,
                        groups: 1,
                        mode: TcuComputeMode.DwConv2d),
                      EAction.TcuDmConfOf(// todo 这里hardcode两个tcu, 后面需要改进
                        TileUtilities.GetTcuIndexBits(tcu_oc / tcu_oc_chunk),
                        is_init_psum ? MathF.Select(MathF.Equal(tcu_oc, glb_of.Region[1].Start), init_psums[0][.., (0, MathF.Min(tcu_oc + tcu_oc_chunk, glb_of.Region[1].Stop) - tcu_oc), .., ..], init_psums[1][.., (0, MathF.Min(tcu_oc + tcu_oc_chunk, glb_of.Region[1].Stop) - tcu_oc), .., ..]) : GlbReIndex(glb_of[.., (tcu_oc, MathF.Min(tcu_oc + tcu_oc_chunk, glb_of.Region[1].Stop)), .., ..], glb_psum, out _),
                        glb_of[.., (tcu_oc, MathF.Min(tcu_oc + tcu_oc_chunk, glb_of.Region[1].Stop)), .., ..],
                        0),
                      EAction.TcuDmConfIf(
                        TileUtilities.GetTcuIndexBits(tcu_oc / tcu_oc_chunk),
                        GlbReIndex(glb_of[.., (tcu_oc, MathF.Min(tcu_oc + tcu_oc_chunk, glb_of.Region[1].Stop)), .., ..], glb_if, out var if_paddings),
                        stride_w: call[GNNEConv2D.Stride][1],
                        stride_h: call[GNNEConv2D.Stride][0],
                        input_c_pre_pu: c_once_tcu,
                        dilation_h: call[GNNEConv2D.Dilation][0],
                        padding_top: if_paddings[2].Before,
                        padding_bottom: if_paddings[2].After,
                        padding_left: if_paddings[3].Before,
                        padding_right: if_paddings[3].After),
                      EAction.TcuDmConfW(
                        TileUtilities.GetTcuIndexBits(tcu_oc / tcu_oc_chunk),
                        GlbReIndex(glb_of[.., (tcu_oc, MathF.Min(tcu_oc + tcu_oc_chunk, glb_of.Region[1].Stop)), .., ..], glb_w, out _)),
                      EAction.TcuDmFetchW(// 4. fetch weights
                        TileUtilities.GetTcuIndexBits(tcu_oc / tcu_oc_chunk),
                        GlbReIndex(glb_of[.., (tcu_oc, MathF.Min(tcu_oc + tcu_oc_chunk, glb_of.Region[1].Stop)), .., ..], glb_w, out _)),
                      EAction.TcuDmFetchIf(// 5. loop over tcus and fetch if for each tcu
                        TileUtilities.GetTcuIndexBits(tcu_oc / tcu_oc_chunk),
                        GlbReIndex(glb_of[.., (tcu_oc, MathF.Min(tcu_oc + tcu_oc_chunk, glb_of.Region[1].Stop)), .., ..], glb_if, out _))))),
                  EAction.TcuPuCompute(// 6. pu compute   NOTE 这里我没有在weight的kh和kw上切,所以默认都是一次算完的
                    TileUtilities.GetNTcuIndexBits(n_active_tcu),
                    true,
                    true,
                    call[GNNEConv2D.PSum] is not Call { Target: Uninitialized },
                    TileUtilities.GetNTcuIndexBits(n_active_tcu)))))))));

        if (promote is null)
        {
            block.Alloc(glb_of.Buffers, is_init_psum ? init_psums[0].Buffers.OfType<Expr>().Concat(init_psums[1].Buffers.Select(b => MathF.Condition(EAction.FoldOfMarker(part_condition > 1), b))).ToArray() : None.Default);
        }
        else if (promote is int promoteIndex)
        {
            if (is_init_psum)
            {
                NestedBlocks[promoteIndex + 1].Alloc(init_psums[0].Buffers.OfType<Expr>().Concat(init_psums[1].Buffers.Select(b => MathF.Condition(EAction.FoldOfMarker(part_condition > 1), b))).ToArray());
            }
        }

        return block;
    }

    protected virtual BufferRegionView[] GetInitPSumBufferRegion(Call call, IndexMapKey key, BufferRegionView glb_psum, int? promote, string prefix, int split_axis, int tcuActNum, out Expr part_condition)
    {
        var chunk = TileUtilities.Split(glb_psum.Dimensions[split_axis], tcuActNum);
        part_condition = TileUtilities.SplitTimes(glb_psum.Dimensions[split_axis], chunk);
        var views = new BufferRegionView[2];
        var psum_dimensions = glb_psum.Dimensions.ToArray();
        psum_dimensions[split_axis] = chunk;
        var dimensions = psum_dimensions;

        // build psum a and b
        foreach (var (part, i) in new[] { "_a", "_b" }.Select((p, i) => (p, i)))
        {
            var name = prefix + "." + TileNames.InitPSum + part;
            var glb_init_psums = new List<Buffer>();
            if (TileOptions.PingPong)
            {
                for (int p = 0; p < TileOptions.PingPongNum; p++)
                {
                    glb_init_psums.Add(new LogicalBuffer(name + $"(p{p})", DataTypes.Float32, MemoryLocation.L2Data, dimensions));
                }
            }
            else
            {
                glb_init_psums.Add(new LogicalBuffer(name, DataTypes.Float32, MemoryLocation.L2Data, dimensions));
            }

            Expr loopCount;
            if (promote is int promoteInt)
            {
                if (promoteInt != -1)
                {
                    loopCount = K510TIRExtensions.PromotedLoopCount(promoteInt, LoopVars, LoopDomains);
                }
                else
                {
                    loopCount = LoopCount;
                }
            }
            else
            {
                loopCount = LoopCount;
            }

            views[i] = new BufferRegionView(glb_init_psums, glb_psum.Bounds, psum_dimensions.Select(d => new Range(0, d, 1)), key, loopCount, promote);
        }

        return views;
    }

    protected virtual Expr GNNEConv2DComputeActEnable(Call call, BufferRegionView glb_w, Expr khStop, Expr kHBounds, Expr kwStop, Expr kWBounds)
    {
        return MathF.LogicalAnd(MathF.GreaterEqual(khStop, kHBounds), MathF.GreaterEqual(kwStop, kWBounds));
    }

    protected virtual Expr GNNEConv2DComputeOfEnable(Call call, BufferRegionView glb_w, Expr khStop, Expr kHBounds, Expr kwStop, Expr kWBounds)
    {
        return GNNEConv2DComputeActEnable(call, glb_w, khStop, kHBounds, kwStop, kWBounds);
    }

    protected virtual Expr GNNEConv2DComputeLoadPsumEnable(Call call, BufferRegionView glb_w, Expr kh, Expr kw)
    {
        if (call[IR.K510.GNNEConv2D.PSum] is Call { Target: IR.Buffers.Uninitialized })
        {
            return IR.F.Math.LogicalNot(IR.F.Math.LogicalAnd(IR.F.Math.Equal(kh, 0), IR.F.Math.Equal(kw, 0)));
        }

        return true;
    }

    /// <summary>
    /// share if 是每个tcu计算一半的oc, 此时他们共享同一个if.
    /// </summary>
    protected ITileBlockBuilder GNNEConv2DSharedIF(Call call, string block_name, BufferRegionView glb_w, BufferRegionView glb_if, BufferRegionView glb_act, BufferRegionView glb_psum, BufferRegionView glb_of, bool is_init_psum, string prefix, int? promote = null)
    {
        var reGlbIf = GlbReIndex(glb_of, glb_if, out var sub_paddings);

        var init_psums = GetInitPSumBufferRegion(call, IndexMapKey.Create(call, GNNEConv2D.PSum), glb_psum, promote, prefix, 1, ExtCompilerServices.Env.TcuActNum, out var part_condition);

        var block = EAction.TileBlock(block_name).Reads(glb_w.BufferRegions, is_init_psum ? init_psums[0].BufferRegions.Concat(init_psums[1].BufferRegions.Select(b => MathF.Condition(EAction.FoldOfMarker(part_condition > 1), b))).ToArray() : glb_psum.BufferRegions, glb_if.BufferRegions, glb_act.BufferRegions).Body(
          T.Let(out var khChunck, MathF.Min(glb_w.Dimensions[2], ExtCompilerServices.Env.PuHeight)).Body(
          T.Let(out var kwChunck, MathF.Min(glb_w.Dimensions[3], ExtCompilerServices.Env.PuKernelSpad)).Body(
          T.Unrolled(out var kh, new(0, glb_w.Dimensions[2], khChunck)).Body(// 对kernel h/w进行tiling 暂时先不考虑
          T.Unrolled(out var kw, new(0, glb_w.Dimensions[3], kwChunck)).Body(
              T.Let(out var tcu_oc_chunk, TileUtilities.Split(glb_of.RegionSize(1), ExtCompilerServices.Env.TcuActNum)).Body(// 1. determine tcu act num
              T.Let(out var n_active_tcu, TileUtilities.SplitTimes(glb_of.RegionSize(1), tcu_oc_chunk)).Body(
                  T.If(MathF.Equal(n_active_tcu, 1)).Then(// 3. broadcast action
                  EAction.TcuDmBroadCast(TcuDivideStrategy.NoShare)).Else(
                  EAction.TcuDmBroadCast(TcuDivideStrategy.ShareIf)),
                  EAction.TcuDmConfIf(// 4. conf if
                  TileUtilities.GetNTcuIndexBits(n_active_tcu),
                  reGlbIf,
                  stride_w: call[GNNEConv2D.Stride][1],
                  stride_h: call[GNNEConv2D.Stride][0],
                  input_c_pre_pu: MathF.Min(ExtCompilerServices.Env.PuHeight / glb_w.RegionSize(2), glb_if.RegionSize(1)), // todo 这里可能有问题.
                  dilation_h: call[GNNEConv2D.Dilation][0],
                  padding_top: sub_paddings[2].Before,
                  padding_bottom: sub_paddings[2].After,
                  padding_left: sub_paddings[3].Before,
                  padding_right: sub_paddings[3].After),
                  T.Unrolled(out var tcu_oc, new(glb_of.Region[1].Start, glb_of.Region[1].Stop, tcu_oc_chunk)).Body(// 5. loop over tcus and config each tcu
                  EAction.TcuPuConfAct(
                    TileUtilities.GetTcuIndexBits(tcu_oc / tcu_oc_chunk),
                    GlbReIndex(glb_of[.., (tcu_oc, MathF.Min(tcu_oc + tcu_oc_chunk, glb_of.Region[1].Stop)), .., ..], glb_act, out _),
                    call[GNNEConv2D.FusedClamp][0],
                    call[GNNEConv2D.FusedClamp][1]),
                  EAction.TcuPuConf(
                    TileUtilities.GetTcuIndexBits(tcu_oc / tcu_oc_chunk),
                    reGlbIf, // 切oc对于if不影响
                    glb_of[.., (tcu_oc, MathF.Min(tcu_oc + tcu_oc_chunk, glb_of.Region[1].Stop)), .., ..],
                    khChunck,
                    kwChunck,
                    m_once: MathF.Min(ExtCompilerServices.Env.PuWidth, MathF.Min(tcu_oc + tcu_oc_chunk, glb_of.Region[1].Stop) - tcu_oc),
                    c_once: MathF.Min(MathF.Min(ExtCompilerServices.Env.PuHeight / glb_w.RegionSize(2), glb_w.RegionSize(1)), glb_if.RegionSize(1)),
                    groups: call[GNNEConv2D.Groups],
                    mode: TcuComputeMode.NormalConv2d),
                  EAction.TcuDmConfOf(// todo 这里hardcode两个tcu, 后面需要改进
                    TileUtilities.GetTcuIndexBits(tcu_oc / tcu_oc_chunk),
                    is_init_psum ? MathF.Select(MathF.Equal(tcu_oc, glb_of.Region[1].Start), init_psums[0][.., (0, MathF.Min(tcu_oc + tcu_oc_chunk, glb_of.Region[1].Stop) - tcu_oc), .., ..], init_psums[1][.., (0, MathF.Min(tcu_oc + tcu_oc_chunk, glb_of.Region[1].Stop) - tcu_oc), .., ..]) : GlbReIndex(glb_of[.., (tcu_oc, MathF.Min(tcu_oc + tcu_oc_chunk, glb_of.Region[1].Stop)), .., ..], glb_psum, out _),
                    glb_of[.., (tcu_oc, MathF.Min(tcu_oc + tcu_oc_chunk, glb_of.Region[1].Stop)), .., ..],
                    0),
                  EAction.TcuDmConfW(
                    TileUtilities.GetTcuIndexBits(tcu_oc / tcu_oc_chunk),
                    GlbReIndex(glb_of[.., (tcu_oc, MathF.Min(tcu_oc + tcu_oc_chunk, glb_of.Region[1].Stop)), .., ..], glb_w, out _))),
                  T.Unrolled(out var tcu_oc2, new(glb_of.Region[1].Start, glb_of.Region[1].Stop, tcu_oc_chunk)).Body(
                    EAction.TcuDmFetchW(// 6. fetch weights.
                      TileUtilities.GetTcuIndexBits(tcu_oc2 / tcu_oc_chunk),
                      GlbReIndex(glb_of[.., (tcu_oc2, MathF.Min(tcu_oc2 + tcu_oc_chunk, glb_of.Region[1].Stop)), .., ..], glb_w, out _))),
                  EAction.TcuDmFetchIf(// 7. fetch if.
                      TileUtilities.GetNTcuIndexBits(n_active_tcu),
                      reGlbIf),
                  EAction.TcuPuCompute(// 8. tcu compute
                      TileUtilities.GetNTcuIndexBits(n_active_tcu),
                      act_enable: GNNEConv2DComputeActEnable(call, glb_w, kh + khChunck, glb_w.Dimensions[2], kw + kwChunck, glb_w.Dimensions[3]),
                      of_enable: GNNEConv2DComputeOfEnable(call, glb_w, kh + khChunck, glb_w.Dimensions[2], kw + kwChunck, glb_w.Dimensions[3]),
                      load_psum: GNNEConv2DComputeLoadPsumEnable(call, glb_w, kh, kw),
                      TileUtilities.GetNTcuIndexBits(n_active_tcu)))))))));

        if (promote is null)
        {
            block.Alloc(glb_of.Buffers, is_init_psum ? init_psums[0].Buffers.OfType<Expr>().Concat(init_psums[1].Buffers.Select(b => MathF.Condition(EAction.FoldOfMarker(part_condition > 1), b))).ToArray() : None.Default);
        }
        else if (promote is int promoteIndex)
        {
            if (is_init_psum)
            {
                NestedBlocks[promoteIndex + 1].Alloc(init_psums[0].Buffers.OfType<Expr>().Concat(init_psums[1].Buffers.Select(b => MathF.Condition(EAction.FoldOfMarker(part_condition > 1), b))).ToArray());
            }
        }

        return block;
    }

    /// <summary>
    /// 假设oc为32被拆分之后每个tcu只能映射一半, 那么两个tcu共享一份weights, 在ofmap的h上进行切分.
    /// </summary>
    protected ITileBlockBuilder GNNEConv2DSharedW(Call call, string block_name, BufferRegionView glb_w, BufferRegionView glb_if, BufferRegionView glb_act, BufferRegionView glb_psum, BufferRegionView glb_of, bool is_depthwise, bool is_init_psum, string prefix, int? promote = null)
    {
        var init_psums = GetInitPSumBufferRegion(call, IndexMapKey.Create(call, GNNEConv2D.PSum), glb_psum, promote, prefix, 2, ExtCompilerServices.Env.TcuActNum, out var part_condition);

        var reGlbW = GlbReIndex(glb_of[.., .., .., ..], glb_w, out _);
        var (iH, iW) = (glb_if.Dimensions[2], glb_if.Dimensions[3]);
        var (kH, kW) = (glb_w.Dimensions[2], glb_w.Dimensions[3]);
        var stride = ((TensorConst)call[IR.K510.GNNEConv2D.Stride]).Value.Cast<int>();
        var padding = ((TensorConst)call[IR.K510.GNNEConv2D.Padding]).Value.Cast<int>();
        var dilation = ((TensorConst)call[IR.K510.GNNEConv2D.Dilation]).Value.Cast<int>();

        var block = EAction.TileBlock(block_name).Reads(glb_w.BufferRegions, is_init_psum ? init_psums[0].BufferRegions.Concat(init_psums[1].BufferRegions.Select(b => MathF.Condition(EAction.FoldOfMarker(part_condition > 1), b))).ToArray() : glb_psum.BufferRegions, glb_if.BufferRegions, glb_act.BufferRegions).Body(
        T.Let(out var khChunck, MathF.Min(kH, ExtCompilerServices.Env.PuHeight)).Body(
        T.Let(out var kwChunck, dilation[1] != 1 ? 1 : MathF.Min(kW, ExtCompilerServices.Env.PuKernelSpad)).Body(
        T.Unrolled(out var kh, new(0, kH, khChunck)).Body(
          T.Unrolled(out var kw, new(0, kW, kwChunck)).Body(// NOTE dw卷积时m once指的是一次ic对应输出多少个oc, 所以默认为1
            T.Let(out var m_once, is_depthwise ? 1 : MathF.Min(ExtCompilerServices.Env.PuWidth, glb_w.RegionSize(0))).Body(
            T.Let(out var c_once, is_depthwise ? MathF.Min(MathF.Min(ExtCompilerServices.Env.PuWidth / m_once, ExtCompilerServices.Env.PuHeight / glb_w.RegionSize(2)), glb_w.RegionSize(0)) : MathF.Min(MathF.Min(ExtCompilerServices.Env.PuHeight / glb_w.RegionSize(2), glb_w.RegionSize(1)), glb_if.RegionSize(1))).Body(// NOTE dw卷积时, if是按对角线排列的, 所以要小于min(pu w/pu h)
            T.Let(out var tcu_oh_chunk, TileUtilities.Split(glb_of.RegionSize(2), ExtCompilerServices.Env.TcuActNum)).Body(// segment tcu h in output_h
            T.Let(out var n_active_tcu, TileUtilities.SplitTimes(glb_of.RegionSize(2), tcu_oh_chunk)).Body(
              T.If(MathF.Equal(n_active_tcu, 1)).Then(// NOTE 这里的psum已经被load好了, 可能到时候会存在psum大小和后续不匹配的问题.// 3. broadcast action
                EAction.TcuDmBroadCast(TcuDivideStrategy.NoShare))
              .Else(
                EAction.TcuDmBroadCast(TcuDivideStrategy.ShareW)),
              EAction.TcuDmConfW(TileUtilities.GetNTcuIndexBits(n_active_tcu), reGlbW[.., .., (kh, MathF.Min(kh + khChunck, kH)), (kw, MathF.Min(kw + kwChunck, kW))]),
              T.Unrolled(out var tcu_oh, new(glb_of.Region[2].Start, glb_of.Region[2].Stop, tcu_oh_chunk)).Body(// 4. conf_w action
                T.Let(out var tcu_index_bits, TileUtilities.GetTcuIndexBits(tcu_oh / tcu_oh_chunk)).Body(// 5. loop over tcus and config each tcu
                  EAction.TcuDmConfIf(// conf if
                    tcu_index_bits,
                    GlbReIndex(glb_of[.., .., (tcu_oh, MathF.Min(tcu_oh + tcu_oh_chunk, glb_of.Region[2].Stop)), ..], glb_if, out var if_padding, (2, r => TileUtilities.Conv2DSubSlice(r, new TIR.Range(kh, IR.F.Math.Min(kh + khChunck, kH), 1), stride[0], padding[0, 0], dilation[0])), (3, r => TileUtilities.Conv2DSubSlice(r, new TIR.Range(kw, IR.F.Math.Min(kw + kwChunck, kW), 1), stride[1], padding[1, 0], dilation[1]))),
                    stride_w: stride[1],
                    stride_h: stride[0],
                    input_c_pre_pu: MathF.Min(ExtCompilerServices.Env.PuHeight / glb_w.RegionSize(2), glb_if.RegionSize(1)), // todo 这里可能有问题.
                    dilation_h: call[GNNEConv2D.Dilation][0],
                    padding_top: if_padding[2].Before,
                    padding_bottom: if_padding[2].After,
                    padding_left: if_padding[3].Before,
                    padding_right: if_padding[3].After),
                  EAction.TcuPuConfAct(// conf act
                    tcu_index_bits,
                    GlbReIndex(glb_of[.., .., (tcu_oh, MathF.Min(tcu_oh + tcu_oh_chunk, glb_of.Region[2].Stop)), ..], glb_act, out _),
                    call[GNNEConv2D.FusedClamp][0],
                    call[GNNEConv2D.FusedClamp][1]),
                  EAction.TcuPuConf(// conf pu
                    tcu_index_bits,
                    GlbReIndex(glb_of[.., .., (tcu_oh, MathF.Min(tcu_oh + tcu_oh_chunk, glb_of.Region[2].Stop)), ..], glb_if, out var _, (2, r => TileUtilities.Conv2DSubSlice(r, new TIR.Range(kh, IR.F.Math.Min(kh + khChunck, kH), 1), stride[0], padding[0, 0], dilation[0])), (3, r => TileUtilities.Conv2DSubSlice(r, new TIR.Range(kw, IR.F.Math.Min(kw + kwChunck, kW), 1), stride[1], padding[1, 0], dilation[1]))),
                    glb_of[.., .., (tcu_oh, MathF.Min(tcu_oh + tcu_oh_chunk, glb_of.Region[2].Stop)), ..],
                    khChunck,
                    kwChunck,
                    m_once,
                    c_once,
                    groups: is_depthwise ? 1 : call[GNNEConv2D.Groups], // NOTE tcu pu conf 的group其实是multiplier的意思，就是一个ic会输出多个oc, 并不是标准conv定义的groups.
                    mode: is_depthwise ? TcuComputeMode.DwConv2d : TcuComputeMode.NormalConv2d),
                  EAction.TcuDmConfOf(// conf of
                    tcu_index_bits,
                    is_init_psum ? MathF.Select(MathF.Equal(tcu_oh, glb_of.Region[2].Start), init_psums[0][.., .., (0, MathF.Min(tcu_oh + tcu_oh_chunk, glb_of.Region[2].Stop) - tcu_oh), .., ..], init_psums[1][.., .., (0, MathF.Min(tcu_oh + tcu_oh_chunk, glb_of.Region[2].Stop) - tcu_oh), .., ..]) : GlbReIndex(glb_of[.., .., (tcu_oh, MathF.Min(tcu_oh + tcu_oh_chunk, glb_of.Region[2].Stop)), ..], glb_psum, out _),
                    glb_of[.., .., (tcu_oh, MathF.Min(tcu_oh + tcu_oh_chunk, glb_of.Region[2].Stop)), ..],
                    0))),
              EAction.TcuDmFetchW(TileUtilities.GetNTcuIndexBits(n_active_tcu), reGlbW[.., .., (kh, MathF.Min(kh + khChunck, kH)), (kw, MathF.Min(kw + kwChunck, kW))]),
              T.Unrolled(out var tcu_oh2, new(glb_of.Region[2].Start, glb_of.Region[2].Stop, tcu_oh_chunk)).Body(// 6. fetch weights
                EAction.TcuDmFetchIf(
                  TileUtilities.GetTcuIndexBits(tcu_oh2 / tcu_oh_chunk), // 7. loop over tcus and fetch if for each tcu
                  GlbReIndex(glb_of[.., .., (tcu_oh2, MathF.Min(tcu_oh2 + tcu_oh_chunk, glb_of.Region[2].Stop)), ..], glb_if, out _, (2, r => TileUtilities.Conv2DSubSlice(r, new TIR.Range(kh, IR.F.Math.Min(kh + khChunck, kH), 1), stride[0], padding[0, 0], dilation[0])), (3, r => TileUtilities.Conv2DSubSlice(r, new TIR.Range(kw, IR.F.Math.Min(kw + kwChunck, kW), 1), stride[1], padding[1, 0], dilation[1]))))),
              EAction.TcuPuCompute(// 8. tcu compute.
                TileUtilities.GetNTcuIndexBits(n_active_tcu),
                GNNEConv2DComputeOfEnable(call, glb_w, kh + khChunck, kH, kw + kwChunck, kW),
                GNNEConv2DComputeActEnable(call, glb_w, kh + khChunck, kH, kw + kwChunck, kW),
                GNNEConv2DComputeLoadPsumEnable(call, glb_w, kh, kw),
                TileUtilities.GetNTcuIndexBits(n_active_tcu)))))))))));

        if (promote is null)
        {
            block.Alloc(glb_of.Buffers, is_init_psum ? init_psums[0].Buffers.OfType<Expr>().Concat(init_psums[1].Buffers.Select(b => MathF.Condition(EAction.FoldOfMarker(part_condition > 1), b))).ToArray() : None.Default);
        }
        else if (promote is int promoteIndex)
        {
            if (is_init_psum)
            {
                NestedBlocks[promoteIndex + 1].Alloc(init_psums[0].Buffers.OfType<Expr>().Concat(init_psums[1].Buffers.Select(b => MathF.Condition(EAction.FoldOfMarker(part_condition > 1), b))).ToArray());
            }
        }

        return block;
    }

    protected virtual ISequentialBuilder<Sequential> LowerGnneConv2D(IndexMapKey parentKey, Call call, GNNEConv2D op, string block_name, string prefix)
    {
        prefix = NameAllocator.Get(nameof(GNNEConv2D));
        bool is_depthwise;
        {
            var groups = ((TensorConst)call[GNNEConv2D.Groups]).Value.ToScalar<int>();
            var input_channels = call[GNNEConv2D.Input].CheckedShape[1].FixedValue;
            var output_channels = call.CheckedShape[1].FixedValue;
            is_depthwise = input_channels == output_channels && output_channels == groups && groups != 1;
        }

        if (is_depthwise)
        {
            prefix = prefix + "(dw)";
            block_name += "(dw)";
        }

        var call_w = IndexMapKey.Create(call, GNNEConv2D.Weights);
        var call_in = IndexMapKey.Create(call, GNNEConv2D.Input);
        var call_act = IndexMapKey.Create(call, GNNEConv2D.Act);
        var call_psum = IndexMapKey.Create(call, GNNEConv2D.PSum);

        bool is_init_psum = call_psum.Expr is Call { Target: Uninitialized };

        TcuDivideStrategy tcu_strategy;
        if (!is_depthwise)
        {
            // 优先让每个tcu的width用满
            var out_shape = call.CheckedShape.ToValueArray();
            if (out_shape[1] >= ExtCompilerServices.Env.PuWidth * ExtCompilerServices.Env.TcuActNum)
            {
                tcu_strategy = TcuDivideStrategy.ShareIf;
            }
            else
            {
                tcu_strategy = TcuDivideStrategy.ShareW;
            }
        }
        else
        { // TODO 需要一种量化的方法来决定dw卷积用什么策略.
            tcu_strategy = TcuDivideStrategy.NoShare;
        }

        prefix = prefix + "." + tcu_strategy;

        // 默认是layer group的做法, 也就是w/act全部promote
        return T.Sequential().Body(
          Visit(call_w, prefix, -1),
          Visit(call_in, prefix),
          Visit(call_act, prefix, -1),
          Visit(call_psum, prefix),
          GetBufferRegion(call_w, out var glb_w),
          GetBufferRegion(call_in, out var glb_if), // glb if 存在padding的情况.
          GetBufferRegion(call_act, out var glb_act),
          GetBufferRegion(call_psum, out var glb_psum, TileOptions.PingPong, name: prefix + "." + GNNEConv2D.PSum.Name), // note 这里的pusm申请了但不记录到allocs中,仅用于给psum apart使用.
          GetBufferRegion(parentKey, out var glb_of, TileOptions.PingPong, name: prefix + "." + TileNames.Output),
          tcu_strategy switch { TcuDivideStrategy.ShareIf => GNNEConv2DSharedIF(call, block_name, glb_w, glb_if, glb_act, glb_psum, glb_of, is_init_psum, prefix), TcuDivideStrategy.ShareW => GNNEConv2DSharedW(call, block_name, glb_w, glb_if, glb_act, glb_psum, glb_of, is_depthwise, is_init_psum, prefix), TcuDivideStrategy.NoShare => GNNEConv2DSharedNone(call, block_name, glb_w, glb_if, glb_act, glb_psum, glb_of, is_init_psum, prefix), _ => throw new NotSupportedException(), });
    }

    protected virtual ISequentialBuilder<Sequential> LowerGnneTranspose(IndexMapKey parentKey, Call call, GNNETranspose op, string block_name, string prefix)
    {
        prefix = NameAllocator.Get(nameof(GNNETranspose));
        var call_in = IndexMapKey.Create(call, GNNETranspose.Input);
        var seq = T.Sequential().Body(
          Visit(call_in, prefix), GetBufferRegion(call_in, out var glb_trans_input), GetBufferRegion(parentKey, out var glb_trans_output, TileOptions.PingPong, name: prefix), EAction.TileBlock(block_name).Alloc(glb_trans_output.Buffers).Reads(glb_trans_input.BufferRegions).Body(EAction.MfuTranspose(glb_trans_input, glb_trans_output, op.Perm)));

        return seq;
    }

    protected virtual ISequentialBuilder<Sequential> LowerGnneCrop(IndexMapKey parentKey, Call call, GNNECrop op, string block_name, string prefix)
    {
        prefix = NameAllocator.Get(nameof(GNNECrop));
        var call_in = IndexMapKey.Create(call, GNNECrop.Input);
        var call_in_bbox = IndexMapKey.Create(call, GNNECrop.InputBBox);
        var seq = T.Sequential().Body(
          Visit(call_in, prefix),
          Visit(call_in_bbox, prefix),
          GetBufferRegion(call_in, out var glb_crop_input),
          GetBufferRegion(call_in_bbox, out var glb_crop_bbox),
          GetBufferRegion(parentKey, out var glb_crop_output, TileOptions.PingPong, name: prefix),
          EAction.TileBlock(block_name).Alloc(glb_crop_output.Buffers).
            Reads(glb_crop_input.BufferRegions, glb_crop_bbox.BufferRegions).
            Body(
              EAction.MfuCrop(
                glb_crop_input,
                glb_crop_output,
                glb_crop_bbox,
                op.ResizeMethod,
                op.AlignMethod,
                op.HalfPixelCenters)));

        return seq;
    }

    protected virtual ISequentialBuilder<Sequential> LowerGnneActivation(IndexMapKey parentKey, Call call, GNNEActivation op, string block_name, string prefix)
    {
        prefix = NameAllocator.Get(nameof(GNNEActivation));
        var fusedclamps = ((TensorConst)call[GNNEActivation.FusedClamp]).Value.Cast<BFloat16>();
        var call_in = IndexMapKey.Create(call, GNNEActivation.Input);
        var call_in_act = IndexMapKey.Create(call, GNNEActivation.Act);

        var seq = T.Sequential().Body(
          Visit(call_in, prefix),
          Visit(call_in_act, prefix),
          GetBufferRegion(call_in, out var glb_if),
          GetBufferRegion(call_in_act, out var glb_act),
          GetBufferRegion(parentKey, out var glb_of, TileOptions.PingPong, name: prefix),
          EAction.TileBlock(block_name).Alloc(glb_of.Buffers).Reads(glb_if.BufferRegions, glb_act.BufferRegions).Body(
            T.Let(out var m_once, 1).Body(
            T.Let(out var c_once, MathF.Min(glb_if.RegionSize(1), ExtCompilerServices.Env.TcuActNum)).Body(
            T.Let(out var tcu_h_chunk, TileUtilities.Split(glb_of.RegionSize(2), ExtCompilerServices.Env.TcuActNum)).Body(// segment tcu h in output_h
            T.Let(out var n_active_tcu, TileUtilities.SplitTimes(glb_of.RegionSize(2), tcu_h_chunk)).Body(
              T.Unrolled(out var tcu_oh, new(glb_of.Region[2].Start, glb_of.Region[2].Stop, tcu_h_chunk)).Body(
                T.Let(out var tcu_index_bits, TileUtilities.GetTcuIndexBits(tcu_oh / tcu_h_chunk)).Body(
                  EAction.TcuPuConfAct(// 1. conf act
                    tcu_index_bits,
                    glb_act,
                    fusedclamps[0],
                    fusedclamps[1]),
                  EAction.TcuPuConf(
                    tcu_index_bits,
                    glb_if[.., .., (tcu_oh, MathF.Min(tcu_oh + tcu_h_chunk, glb_if.Region[2].Stop)), ..],
                    glb_of[.., .., (tcu_oh, MathF.Min(tcu_oh + tcu_h_chunk, glb_of.Region[2].Stop)), ..],
                    1,
                    1,
                    m_once,
                    c_once,
                    1,
                    TcuComputeMode.Activation),
                  EAction.TcuDmConfOf(
                    tcu_index_bits,
                    glb_if[.., .., (tcu_oh, MathF.Min(tcu_oh + tcu_h_chunk, glb_if.Region[2].Stop)), ..],
                    glb_of[.., .., (tcu_oh, MathF.Min(tcu_oh + tcu_h_chunk, glb_of.Region[2].Stop)), ..],
                    0))),
              EAction.TcuPuComputeDummy(TileUtilities.GetNTcuIndexBits(n_active_tcu), true)))))));
        return seq;
    }

    protected virtual ISequentialBuilder<Sequential> LowerGnnePdpReduce(IndexMapKey parentKey, Call call, GNNEPdpReduce op, string block_name, string prefix)
    {
        prefix = NameAllocator.Get(nameof(GNNEPdpReduce));
        var call_in = IndexMapKey.Create(call, GNNEPdpReduce.Input);

        // var ddr_if = BoundsInferGraph[call_in];
        // GlbReIndex(glb_of[.., .., (tcu_oh, MathF.Min(tcu_oh + tcu_h_chunk, glb_of.Region[2].Stop)), ..], glb_if, out var if_paddings)
        var seq = T.Sequential().Body(
          Visit(call_in, prefix),
          GetBufferRegion(call_in, out var glb_if),
          GetBufferRegion(parentKey, out var glb_of, TileOptions.PingPong, name: prefix));
        GlbReIndex(glb_of, glb_if, out var sub_paddings);
        seq.Body(
          EAction.TileBlock(block_name).Alloc(glb_of.Buffers).Reads(glb_if.BufferRegions).Body(
            EAction.PdpReduce(
              glb_if,
              glb_of,
              call[GNNEPdpReduce.Filter],
              call[GNNEPdpReduce.Stride],
              sub_paddings[2].Before,
              sub_paddings[2].After,
              sub_paddings[3].Before,
              sub_paddings[3].After,
              op.ReduceOp)));
        return seq;
    }

    protected virtual ISequentialBuilder<Sequential> LowerGnneConv2DTranspose(IndexMapKey parentKey, Call call, GNNEConv2DTranspose op, string block_name, string prefix)
    {
        throw new NotSupportedException();
    }
}
#endif
