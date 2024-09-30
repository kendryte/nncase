// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

#define USE_KERNEL_LIB
using System.Linq;
using System.Reactive;
using NetFabric.Hyperlinq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Imaging;
using Nncase.IR.Math;
using Nncase.IR.NN;
using Nncase.IR.Tensors;
using Nncase.PatternMatch;
using Nncase.TIR;
using Nncase.TIR.Builders;
using Nncase.Utilities;
using Buffer = Nncase.TIR.Buffer;

namespace Nncase.Passes.Tile;

internal struct TileScope : IDisposable
{
    private static readonly List<Dictionary<Expr, Buffer>> _bufferMapStack = new();
    private static readonly List<IBlockBuilder> _blockBuilderStack = new();
    private static readonly Stack<TileFrame> _frames = new();
    private static readonly List<List<ISequentialBuilder<For>>> _loopBuildersStack = new();
    private static readonly List<List<Var>> _loopVarsStack = new();

    public TileScope(TileFrame frame)
    {
        _frames.Push(frame);
        frame.Enter();
    }

    public static IBlockBuilder CurrentBlock => _blockBuilderStack.Count == 0 ? null! : _blockBuilderStack[^1];

    public static IReadOnlyDictionary<Expr, Buffer> CurrentMap => _bufferMapStack.Count == 0 ? null! : _bufferMapStack[^1];

    public static IReadOnlyList<Var> CurrentLoopVars => _loopVarsStack.Count == 0 ? null! : _loopVarsStack[^1];

    public static IReadOnlyList<IReadOnlyList<Var>> LoopVarStack => _loopVarsStack;

    public static IReadOnlyList<ISequentialBuilder<For>> CurrentLoops => _loopBuildersStack.Count == 0 ? null! : _loopBuildersStack[^1];

    public void Dispose()
    {
        var frame = _frames.Pop();
        frame.Exit();
    }

    public abstract class TileFrame
    {
        public abstract void Enter();

        public abstract void Exit();
    }

    public sealed class PushMemoryFrame : TileFrame
    {
        private readonly Dictionary<Expr, Buffer> _bufferMap;
        private readonly IBlockBuilder _fusionBlock;
        private readonly ISequentialBuilder<For>[] _builders;
        private readonly Var[] _vars;

        public PushMemoryFrame(Dictionary<Expr, Buffer> bufferMap, IBlockBuilder fusionBlock, ISequentialBuilder<For>[] builders, Var[] vars)
        {
            _bufferMap = bufferMap;
            _fusionBlock = fusionBlock;
            _builders = builders;
            _vars = vars;
        }

        public override void Enter()
        {
            _bufferMapStack.Add(_bufferMap);
            _blockBuilderStack.Add(_fusionBlock);
            _loopBuildersStack.Add(new(_builders));
            _loopVarsStack.Add(new(_vars));
        }

        public override void Exit()
        {
            _bufferMapStack.RemoveAt(_bufferMapStack.Count - 1);
            _blockBuilderStack.RemoveAt(_blockBuilderStack.Count - 1);
            _loopBuildersStack.RemoveAt(_loopBuildersStack.Count - 1);
            _loopVarsStack.RemoveAt(_loopVarsStack.Count - 1);
        }
    }

    public sealed class PushLoopFrame : TileFrame
    {
        private readonly ISequentialBuilder<For>[] _builders;
        private readonly Var[] _vars;

        public PushLoopFrame(ISequentialBuilder<For>[] builders, Var[] vars)
        {
            _builders = builders;
            _vars = vars;
        }

        public override void Enter()
        {
            _loopBuildersStack[^1].AddRange(_builders);
            _loopVarsStack[^1].AddRange(_vars);
        }

        public override void Exit()
        {
            var total = _loopBuildersStack[^1].Count;
            int length = _builders.Length;
            _loopBuildersStack[^1].RemoveRange(total - length, length);
            total = _loopVarsStack[^1].Count;
            length = _vars.Length;
            _loopVarsStack[^1].RemoveRange(total - length, length);
        }
    }
}

internal sealed class DeviceFusionToPrimFuncRewriter : ExprRewriter
{
    private readonly HashSet<PrimFunction> _primFunctions = new(ReferenceEqualityComparer.Instance);
    private readonly IReadOnlyDictionary<Fusion, FusionChecker> _fusionCheckCache;

    public DeviceFusionToPrimFuncRewriter(Dictionary<Fusion, FusionChecker> fusionCheckCache)
    {
        _fusionCheckCache = fusionCheckCache;
    }

    public HashSet<PrimFunction> PrimFunctions => _primFunctions;

    protected override Expr DefaultRewriteLeaf(Expr expr) => base.DefaultRewriteLeaf(expr);

    protected override Expr RewriteLeafFusion(Fusion expr)
    {
        if (expr.ModuleKind == Targets.CPUTarget.Kind && expr.Name.EndsWith("device"))
        {
            // var oldBody = expr.Body;
            // PrimTileVisitor primTileVisitor = new();
            // primTileVisitor.Visit(oldBody);
            // FusionChecker fusionChecker = new(primTileVisitor.TileList, primTileVisitor.NameList);
            // var tileMap = fusionChecker.Check(oldBody)[0];
            if (!_fusionCheckCache.TryGetValue(expr, out var cachedChecker))
            {
                PrimTileVisitor primTileVisitor = new();
                primTileVisitor.Visit(expr.Body);
                cachedChecker = new FusionChecker(primTileVisitor.TileList);
                cachedChecker.Check(expr.Body);
            }

            if (cachedChecker.CheckedResult.Count != 1)
            {
                throw new NotSupportedException("Not support no uniform shard!");
            }

            var (_, tileMap) = cachedChecker.CheckedResult[0];

            // var tileShape = tileMap[oldBody].OutShape;
            // var newBody = IR.F.CPU.Store(
            //     tileShape,
            //     new TileType(TIR.MemoryLocation.Output, DistributedUtility.GetDividedTensorType((DistributedType)oldBody.CheckedType)),
            //     new TileFusionLowerCloner(tileMap).Clone(oldBody, default));

            // var egraph = new EGraph(newBody);
            // CompilerServices.ERewrite(egraph, new IRewriteRule[] { new UnaryL1Fusion(), new MatmulL1Fusion() }, new());
            // var tiledBody = egraph.Extract(egraph.Root!, new TileFusionCostEvaluator(), out var _);
            // var newfusion = new Fusion(expr.Name, Targets.CPUTarget.Kind, tiledBody, expr.Parameters);

            // if (Diagnostics.DumpScope.Current.IsEnabled(Diagnostics.DumpFlags.Tiling))
            // {
            //     Diagnostics.DumpScope.Current.DumpIR(newfusion, string.Empty, "L1Tiled");
            // }

            // var allocMap = fusionChecker.ReAllocate(newfusion.Body, true);
            var converter = new DeviceToTIRConverter(expr, tileMap);
            var primfunc = converter.Convert();
            _primFunctions.Add(primfunc);
            return primfunc;
        }

        return expr;
    }
}

internal sealed class TileFusionCostEvaluator : Evaluator.IBaseFuncCostEvaluator
{
    public Cost VisitLeaf(BaseFunction target)
    {
        return new Cost()
        {
            [CostFactorNames.CPUCycles] = 1000,
        };
    }
}

internal sealed class DeviceToTIRConverter
{
    private readonly Fusion _fusion;
    private readonly IReadOnlyDictionary<Expr, NodeInfo> _tileMemo;
    private readonly Dictionary<Expr, BufferRegion> _regionMemo;

    public DeviceToTIRConverter(Fusion expr, IReadOnlyDictionary<Expr, NodeInfo> tileMap)
    {
        _fusion = expr;
        _tileMemo = tileMap;
        _regionMemo = new(ReferenceEqualityComparer.Instance);
    }

    public TIR.PrimFunction Convert()
    {
        var shape = _fusion.Body.CheckedShape;
        var func = T.PrimFunc(_fusion.Name, Targets.CPUTarget.Kind, _fusion.Parameters.ToArray().Select(p => _tileMemo[p].Buffer).Concat(new[] { _tileMemo[_fusion.Body].Buffer }).ToArray()).Body(
            Visit(_fusion, AffineMap.Identity(shape.Rank), null!, out _));
        return func.Build();
    }

    public Expr Visit(Expr expr, AffineMap rootMap, BufferRegion outRegion, out AffineMap[] inputMaps)
    {
        inputMaps = Array.Empty<AffineMap>();
        return expr switch
        {
            Call call => (call.Target switch
            {
                IR.CPU.Load op => LowerLoad(call, op, rootMap, outRegion, out inputMaps),
                IR.CPU.Store op => LowerStore(call, op, rootMap, outRegion, out inputMaps),
                IR.Math.Unary op => LowerUnary(call, op, rootMap, outRegion, out inputMaps),
                IR.Math.MatMul op => LowerMatmul(call, op, rootMap, outRegion, out inputMaps),
                IR.Math.Binary op => LowerBinary(call, op, rootMap, outRegion, out inputMaps),
                Fusion func => LowerFusion(call, func, rootMap, outRegion, out inputMaps),
                _ => throw new NotSupportedException(),
            }).Build(),
            Fusion func => LowerFusion(null, func, rootMap, outRegion, out inputMaps).Build(),
            _ => T.Nop(),
        };
    }

    private ISequentialBuilder<Sequential> LowerMatmul(Call call, MatMul op, AffineMap rootMap, BufferRegion outRegion, out AffineMap[] inputMaps)
    {
        var lhsTile = GetTile(call.Arguments[0]);
        var lhsShape = GetShape(call.Arguments[0]);
        var rhsShape = GetShape(call.Arguments[1]);
        var rhsTile = GetTile(call.Arguments[1]);
        var tileShape = GetTile(call);
        var fullShape = GetShape(call);

        Expr[] PostProcessAffineMap(List<Expr> iters, IReadOnlyList<int> inShape, IReadOnlyList<int> outShape)
        {
            var ralign = outShape.Count - inShape.Count;
            for (int i = outShape.Count - 1; i >= 0; i--)
            {
                if (i < ralign)
                {
                    iters.RemoveAt(i);
                }
                else if (i < (outShape.Count - 2) && inShape[i] == 1 && outShape[i] != 1)
                {
                    iters[i] = 0;
                }
            }

            return iters.ToArray();
        }

        var outKLoop = T.ForLoop(out var ok, new TIR.Range(0, lhsShape[^1], lhsTile[^1]), LoopMode.Serial);
        using (new TileScope(new TileScope.PushLoopFrame(new[] { outKLoop }, new[] { ok })))
        {
            Expr[] LhsFunc(params Expr[] exprs)
            {
                return PostProcessAffineMap(exprs[..^2].Concat(new[] { exprs[^1] }).ToList(), lhsShape, fullShape);
            }

            Expr[] RhsFunc(params Expr[] exprs)
            {
                return PostProcessAffineMap(exprs[..^3].Concat(new[] { exprs[^1], exprs[^2] }).ToList(), rhsShape, fullShape);
            }

            var lhsMap = AffineMap.FromCallable(LhsFunc, fullShape.Count, 1).Compose(rootMap);
            var rhsMap = AffineMap.FromCallable(RhsFunc, fullShape.Count, 1).Compose(rootMap);

            var outStarts = outRegion.Region.ToArray().Select(r => r.Start).ToList();
            outStarts.Add(0);
            var outStops = outRegion.Region.ToArray().Select(r => r.Stop).ToList();
            outStops.Add(IR.F.Math.Min(ok + lhsTile[^1], lhsShape[^1]) - ok);

            var lhsRegion = GetBufferRegion(call.Arguments[0], (TIR.Buffer lhsBuffer) =>
            {
                var lhsStarts = lhsMap.Apply(outStarts.ToArray());
                var lhsStops = lhsMap.Apply(outStops.ToArray());
                return new BufferRegion(lhsBuffer, lhsStarts.Zip(lhsStops).Select(p => new TIR.Range(p.First, p.Second, 1)).ToArray());
            });

            var rhsRegion = GetBufferRegion(call.Arguments[1], (TIR.Buffer rhsBuffer) =>
            {
                var rhsStarts = rhsMap.Apply(outStarts.ToArray());
                var rhsStops = rhsMap.Apply(outStops.ToArray());
                return new BufferRegion(rhsBuffer, rhsStarts.Zip(rhsStops).Select(p => new TIR.Range(p.First, p.Second, 1)).ToArray());
            });
            TileScope.CurrentBlock.Alloc(outRegion.Buffer);
            var block = T.Block(nameof(MatMul)).
                    Reads(lhsRegion, rhsRegion).
                    Writes(outRegion);
            outKLoop.Body(
                Visit(call.Arguments[0], lhsMap, lhsRegion, out var lhsInputMaps),
                Visit(call.Arguments[1], rhsMap, rhsRegion, out var rhsInputMaps),
                block);
#if USE_KERNEL_LIB
            block.Body(TIR.F.CPU.Matmul(lhsRegion, rhsRegion, outRegion, None.Default));
#else
            // var lhsStarts = lhsRegion.Region.ToArray().Select(r => (T.Let(out var start, r.Start), start)).ToArray();
            // var rhsStarts = rhsRegion.Region.ToArray().Select(r => (T.Let(out var start, r.Start), start)).ToArray();
            // var outLetStarts = outStarts.ToArray().Select(r => (T.Let(out var start, r), start)).ToArray();
            var stopLets = outStops.Select((s, i) => (T.Let(out var stop, s, $"stop{i}"), stop)).ToArray();
            var compute = T.Grid(out var vars, LoopMode.Serial, stopLets.Select((p, i) => new TIR.Range(0, p.stop, i < stopLets.Length - 3 ? 1 : 32)).ToArray()).Body(
                T.Let(out var curM, IR.F.Math.Min(stopLets[^3].stop - vars[^3], 32)).Body(
                T.Let(out var curN, IR.F.Math.Min(stopLets[^2].stop - vars[^2], 32)).Body(
                T.Let(out var curK, IR.F.Math.Min(stopLets[^1].stop - vars[^1], 32)).Body(
                    TIR.F.CPU.TMMA(
                        GetBufferPtr(lhsRegion, lhsMap.Apply(vars).Select((v, i) => v + lhsRegion.Region[i].Start).ToArray()),
                        GetBufferPtr(rhsRegion, rhsMap.Apply(vars).Select((v, i) => v + rhsRegion.Region[i].Start).ToArray()),
                        GetBufferPtr(outRegion, vars.SkipLast(1).Select((v, i) => v + outRegion.Region[i].Start).ToArray()),
                        curM,
                        curK,
                        curN,
                        lhsRegion.Buffer.Strides[^2],
                        rhsRegion.Buffer.Strides[^2],
                        outRegion.Buffer.Strides[^2],
                        DataTypes.Float32,
                        lhsRegion.Buffer.ElemType,
                        outRegion.Buffer.ElemType,
                        IR.F.Math.NotEqual(vars[^1] + ok, 0))))));

            var final = stopLets.Select(p => p.Item1).Aggregate((acc, cur) =>
            {
                acc.Body(cur);
                return cur;
            });
            final.Body(compute);
            block.Body(stopLets[0].Item1);
#endif
        }

        // var fullK = ((TileType)call.Arguments[0].CheckedType).TensorType.Shape[^1].FixedValue;
        Expr[] LhsInFunc(params Expr[] exprs) => PostProcessAffineMap(exprs[..^1].Concat(new Expr[] { 0 }).ToList(), lhsShape, fullShape);
        Expr[] RhsInFunc(params Expr[] exprs) => PostProcessAffineMap(exprs[..^2].Concat(new Expr[] { 0, exprs[^1] }).ToList(), rhsShape, fullShape);

        // root = (b,c,m,n) -> (b,c,m,n)
        // lhs loop vars = b,c,m,k
        inputMaps = new[] {
            AffineMap.FromCallable(LhsInFunc, fullShape.Count, 0).Compose(rootMap),
            AffineMap.FromCallable(RhsInFunc, fullShape.Count, 0).Compose(rootMap),
        };

        return T.Sequential().Body(outKLoop);
    }

    private ISequentialBuilder<Sequential> LowerLoad(Call call, IR.CPU.Load load, AffineMap rootMap, BufferRegion outRegion, out AffineMap[] inputMaps)
    {
        var tileShape = GetTile(call);
        var inShape = GetShape(call.Arguments[0]);
        var iterVars = rootMap.Apply(TileScope.CurrentLoopVars.ToArray());
        inputMaps = new[] { rootMap };

        var inRegion = GetBufferRegion(call.Arguments[0], (TIR.Buffer inBuffer) =>
           new BufferRegion(inBuffer, Enumerable.Range(0, tileShape.Count).Select(i =>
           {
               var iterV = iterVars[i];
               return new TIR.Range(iterV, IR.F.Math.Min(iterV + tileShape[i], inShape[i]), 1);
           }).ToArray()));
        TileScope.CurrentBlock.Alloc(outRegion.Buffer);
        var block = T.Block("load").
                Reads(inRegion).
                Writes(outRegion);
        var seq = T.Sequential().Body(
            Visit(call.Arguments[0], rootMap, inRegion, out var _),
            block);
#if USE_KERNEL_LIB
        block.Body(T.Memcopy(outRegion, inRegion));
#else
        // var inStarts = inRegion.Region.ToArray().Select(r => (T.Let(out var start, r.Start), start)).ToArray();
        // var outStarts = outRegion.Region.ToArray().Select(r => (T.Let(out var start, r.Start), start)).ToArray();
        var compute = T.Grid(out var vars, LoopMode.Serial, inRegion.Region.ToArray().Select(r => new TIR.Range(0, r.Stop - r.Start, 1)).ToArray()).
            Body(
            T.BufferStore(outRegion.Buffer, vars.Select((v, i) => v + outRegion.Region[i].Start).ToArray(), T.BufferLoad(inRegion.Buffer, vars.Select((v, i) => v + inRegion.Region[i].Start).ToArray())));

        // var final = inStarts.Concat(outStarts).Select(p => p.Item1).Aggregate((acc, cur) =>
        // {
        //     acc.Body(cur);
        //     return cur;
        // });
        // final.Body(compute);
        // block.Body(inStarts[0].Item1);
        block.Body(compute);
#endif

        return seq;
    }

    private ISequentialBuilder<Sequential> LowerStore(Call call, IR.CPU.Store store, AffineMap rootMap, BufferRegion outRegion, out AffineMap[] inputMaps)
    {
        var iterVars = rootMap.Apply(TileScope.CurrentLoopVars.ToArray());
        var tileShape = GetTile(call);
        var outShape = GetShape(call);

        outRegion = GetBufferRegion(call, (TIR.Buffer outBuffer) =>
           new BufferRegion(outBuffer, Enumerable.Range(0, tileShape.Count).Select(i =>
           {
               var iterV = iterVars[i];
               return new TIR.Range(iterV, IR.F.Math.Min(iterV + tileShape[i], outShape[i]), 1);
           }).ToArray()));

        var inRegion = GetBufferRegion(call.Arguments[0], (TIR.Buffer inBuffer) =>
            new BufferRegion(inBuffer, Enumerable.Range(0, tileShape.Count).Select(i =>
            {
                // var iterV = iterVars[i];
                return new TIR.Range(0, outRegion.Region[i].Stop - outRegion.Region[i].Start, 1);
            }).ToArray()));

        var block = T.Block(nameof(store)).
            Reads(inRegion).
            Writes(outRegion);
        var seq = T.Sequential().Body(
            Visit(call.Arguments[0], rootMap, inRegion, out inputMaps),
            block);
#if USE_KERNEL_LIB
        block.Body(T.Memcopy(outRegion, inRegion));
#else
        // var inStarts = inRegion.Region.ToArray().Select(r => (T.Let(out var start, r.Start), start)).ToArray();
        // var outStarts = outRegion.Region.ToArray().Select(r => (T.Let(out var start, r.Start), start)).ToArray();
        var compute = T.Grid(out var vars, LoopMode.Serial, inRegion.Region.ToArray().Select(r => new TIR.Range(0, r.Stop - r.Start, 1)).ToArray()).
            Body(
            T.BufferStore(outRegion.Buffer, vars.Select((v, i) => v + outRegion.Region[i].Start).ToArray(), T.BufferLoad(inRegion.Buffer, vars.Select((v, i) => v + inRegion.Region[i].Start).ToArray())));

        // var final = inStarts.Concat(outStarts).Select(p => p.Item1).Aggregate((acc, cur) =>
        // {
        //     acc.Body(cur);
        //     return cur;
        // });
        // final.Body(compute);
        // block.Body(inStarts[0].Item1);
        block.Body(compute);
#endif
        return seq;
    }

    private ISequentialBuilder<Sequential> LowerBinary(Call call, Binary op, AffineMap rootMap, BufferRegion outRegion, out AffineMap[] inputMaps)
    {
        var lhsShape = GetShape(call.Arguments[0]);
        var rhsShape = GetShape(call.Arguments[1]);
        var fullShape = GetShape(call);
        var lhsRegion = GetBufferRegion(call.Arguments[0], (TIR.Buffer inBuffer) => new BufferRegion(inBuffer, outRegion.Region));
        var rhsRegion = GetBufferRegion(call.Arguments[1], (TIR.Buffer inBuffer) => new BufferRegion(inBuffer, outRegion.Region));
        TileScope.CurrentBlock.Alloc(outRegion.Buffer);

        Expr[] PostProcessAffineMap(List<Expr> iters, IReadOnlyList<int> inShape, IReadOnlyList<int> outShape)
        {
            var ralign = outShape.Count - inShape.Count;
            for (int i = outShape.Count - 1; i >= 0; i--)
            {
                if (i < ralign)
                {
                    iters.RemoveAt(i);
                }
                else if (i < (outShape.Count - 2) && inShape[i] == 1 && outShape[i] != 1)
                {
                    iters[i] = 0;
                }
            }

            return iters.ToArray();
        }

        Expr[] LhsInFunc(params Expr[] exprs) => PostProcessAffineMap(exprs.ToList(), lhsShape, fullShape);
        Expr[] RhsInFunc(params Expr[] exprs) => PostProcessAffineMap(exprs.ToList(), rhsShape, fullShape);

        inputMaps = new[] {
            AffineMap.FromCallable(LhsInFunc, fullShape.Count, 0).Compose(rootMap),
            AffineMap.FromCallable(RhsInFunc, fullShape.Count, 0).Compose(rootMap),
        };

        var block = T.Block("binary").
                Reads(lhsRegion, rhsRegion).
                Writes(outRegion);
        var seq = T.Sequential().Body(
            Visit(call.Arguments[0], rootMap, lhsRegion, out _),
            Visit(call.Arguments[1], rootMap, rhsRegion, out _),
            block);
#if USE_KERNEL_LIB
        block.Body(TIR.F.CPU.Binary(op.BinaryOp, lhsRegion, rhsRegion, outRegion));
#else
    throw new NotSupportedException();
#endif
        return seq;
    }

    private ISequentialBuilder<Sequential> LowerUnary(Call call, Unary op, AffineMap rootMap, BufferRegion outRegion, out AffineMap[] inputMaps)
    {
        // var iterVars = rootMap.Apply(TileScope.CurrentLoopVars.ToArray());
        var inRegion = GetBufferRegion(call.Arguments[0], (TIR.Buffer inBuffer) => new BufferRegion(inBuffer, outRegion.Region));
        TileScope.CurrentBlock.Alloc(outRegion.Buffer);
        inputMaps = new[] { rootMap };
        var block = T.Block("unary").
                Reads(inRegion).
                Writes(outRegion);
        var seq = T.Sequential().Body(
            Visit(call.Arguments[0], rootMap, inRegion, out _),
            block);
#if USE_KERNEL_LIB
        block.Body(TIR.F.CPU.Unary(op.UnaryOp, inRegion, outRegion));
#else
        // var inStarts = inRegion.Region.ToArray().Select(r => (T.Let(out var start, r.Start), start)).ToArray();
        // var outStarts = outRegion.Region.ToArray().Select(r => (T.Let(out var start, r.Start), start)).ToArray();
        var compute = T.Grid(out var vars, LoopMode.Serial, inRegion.Region.ToArray().Select(r => new TIR.Range(0, r.Stop - r.Start, 1)).ToArray()).
            Body(
            T.BufferStore(outRegion.Buffer, vars.Select((v, i) => v + outRegion.Region[i].Start).ToArray(), IR.F.Math.Unary(op.UnaryOp, T.BufferLoad(inRegion.Buffer, vars.Select((v, i) => v + inRegion.Region[i].Start).ToArray()))));

        // var final = inStarts.Concat(outStarts).Select(p => p.Item1).Aggregate((acc, cur) =>
        // {
        //     acc.Body(cur);
        //     return cur;
        // });
        // final.Body(compute);
        // block.Body(inStarts[0].Item1);
        block.Body(compute);
#endif
        return seq;
    }

    private ISequentialBuilder<Sequential> LowerFusion(Call? call, Fusion func, AffineMap rootMap, BufferRegion outRegion, out AffineMap[] inputMaps)
    {
        if (func.Body is not Call { Target: IR.CPU.Store store })
        {
            throw new NotSupportedException();
        }

        // var inBuffer = call is null ? GetBuffer(func.Parameters[0]) : GetBuffer(call.Arguments[0]);
        // var outBuffer = call is null ? GetBuffer(func.Body) : GetBuffer(call);

        // 1. func body
        var fusionBlock = T.Block("main");
        var outShape = GetShape(func.Body);
        var outTile = GetTile(func.Body);
        var nestBuilder = T.Grid(out var loopVars, out var loops, LoopMode.Serial, Enumerable.Range(0, outShape.Count).Select(i => new TIR.Range(0, outShape[i], outTile[i])).ToArray());

        AffineMap[] bodyinputMaps;
        using (new TileScope(
            new TileScope.PushMemoryFrame(
                new Dictionary<Expr, Buffer>(ReferenceEqualityComparer.Instance)
                {
                    // { func.Parameters[0], inBuffer }, { func.Body, outBuffer },
                },
                fusionBlock,
                loops,
                loopVars)))
        {
            fusionBlock.Body(
                nestBuilder.Body(
                    Visit(func.Body, rootMap, outRegion, out bodyinputMaps)));
        }

        var seq = T.Sequential();

        inputMaps = bodyinputMaps;
        if (call is not null)
        {
            for (int i = 0; i < call.Arguments.Length; i++)
            {
                AffineMap[] inmaps = Array.Empty<AffineMap>();
                seq.Body(Visit(call.Arguments[i], bodyinputMaps[i], outRegion, out _));
            }
        }

        // 2. visit args.
        return seq.Body(fusionBlock);
    }

    private TIR.Range[] ComputeRanges(IReadOnlyList<int> tiles, AffineMap rootMap)
    {
        var starts = rootMap.Apply(TileScope.CurrentLoopVars.ToArray());
        return starts.Zip(tiles).Select(p => new TIR.Range(p.First, p.First + p.Second, 1)).ToArray();
    }

    private Expr[] ComputeIndcies(TIR.Buffer top, Expr[] loopvars, AffineMap rootMap)
    {
        var topLevel = top.MemSpan.Location switch
        {
            MemoryLocation.Input or MemoryLocation.Output or MemoryLocation.Rdata => 0,
            MemoryLocation.L2Data => 1,
            _ => throw new InvalidDataException(),
        };

        var newLoopvars = loopvars.ToArray();

        for (int level = TileScope.LoopVarStack.Count - 1; level >= topLevel; level--)
        {
            var mappedVars = rootMap.Apply(TileScope.LoopVarStack[level].ToArray());
            System.Diagnostics.Trace.Assert(mappedVars.Length == newLoopvars.Length);

            for (int i = 0; i < newLoopvars.Length; i++)
            {
                newLoopvars[i] += mappedVars[i];
            }
        }

        return newLoopvars;
    }

    private IReadOnlyList<int> GetTile(Expr expr) => _tileMemo[expr].TileShape;

    private IReadOnlyList<int> GetShape(Expr expr) => _tileMemo[expr].OutShape;

    private BufferRegion GetBufferRegion(Expr expr, Func<TIR.Buffer, TIR.BufferRegion> createFunc)
    {
        var buf = _tileMemo[expr].Buffer;
        if (!_regionMemo.TryGetValue(expr, out var region))
        {
            region = createFunc(buf);
            _regionMemo.Add(expr, region);
        }

        return region;
    }
}
