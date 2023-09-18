// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Reactive;
using System.Runtime.CompilerServices;
using DryIoc.FastExpressionCompiler.LightExpression;
using NetFabric.Hyperlinq;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.Buffers;
using Nncase.IR.CPU;
using Nncase.IR.F;
using Nncase.IR.Math;
using Nncase.IR.NN;
using Nncase.IR.Tensors;
using Nncase.Passes.Mutators;
using Nncase.PatternMatch;
using Nncase.Schedule;
using Nncase.Targets;
using Nncase.TIR;
using Nncase.TIR.Builders;
using Buffer = Nncase.TIR.Buffer;
using MathF = Nncase.IR.F.Math;
using Range = Nncase.TIR.Range;
using Tuple = Nncase.IR.Tuple;

namespace Nncase.Passes.Tile;

/// <summary>
/// convert the fusion to prim func.
/// </summary>
internal sealed class SingleCPUFusionConverter
{
    public SingleCPUFusionConverter(TileOptions tileOptions)
    {
        TileOptions = tileOptions;
    }

    public TileOptions TileOptions { get; }

    public TIR.PrimFunction Convert(Fusion fusion)
    {
        // 1. convert to distribute graph
        var distConverter = new DistributeConvertVisitor(TileOptions);
        var candidatebodys = distConverter.Convert(fusion.Body, out var stagesMap);
        var graph = new EGraph();
        var bodyEclasses = candidatebodys.Select(graph.Add).ToArray();

        // foreach (var (b, i) in candidatebodys.Select((b, i) => (b, i)))
        // {
        //     DumpScope.Current.DumpDotIR(b, "body_" + i.ToString(), "Distribute");
        // }
        // 2. union stages
        foreach (var ((_, equivals), i) in stagesMap.Select((k, i) => (k, i)))
        {
            // foreach (var (b, j) in equivals.Select((b, j) => (b, j)))
            // {
            //     DumpScope.Current.DumpDotIR(b, $"eq_{i}_{j}", "Distribute");
            // }
            var equivalEclasses = equivals.Select(graph.Add).ToArray();

            foreach (var cls in equivalEclasses.Skip(1))
            {
                graph.Union(equivalEclasses[0], cls);
            }
        }

        // 3. union final body
        foreach (var cls in bodyEclasses.Skip(1))
        {
            graph.Union(bodyEclasses[0], cls);
        }

        graph.Rebuild();
        var body = graph.Extract(bodyEclasses[0], null);
        var newFusion = fusion.With(body: body);
        if (DumpScope.Current.IsEnabled(DumpFlags.Tiling))
        {
            DumpScope.Current.DumpDotIR(newFusion, newFusion.Name, "Distribute");
        }

        // 2. convert new fusion to prim func
        var primBody = new List<Expr>();
        var visitor = new ConvertVisitor(primBody);
        visitor.Visit(newFusion);
        return T.PrimFunc(newFusion.Name, newFusion.ModuleKind, visitor.InputBuffers.Concat(visitor.OutputBuffers).ToArray()).Body(primBody.ToArray()).Build();
    }

    private sealed class DistributeConvertVisitor : ExprVisitor<IReadOnlyList<Expr>, Unit>
    {
        private readonly Dictionary<Expr, List<Expr>> _stagesMap;

        public DistributeConvertVisitor(TileOptions tileOptions)
        {
            TileOptions = tileOptions;
            Placement = new Placement(Placement.DeviceKind.CPU, tileOptions.Hierarchy, "bt");
            _stagesMap = new Dictionary<Expr, List<Expr>>(ReferenceEqualityComparer.Instance);
        }

        public TileOptions TileOptions { get; }

        public Placement Placement { get; }

        public IReadOnlyList<Expr> Convert(Expr body, out IReadOnlyDictionary<Expr, List<Expr>> stagesMap)
        {
            stagesMap = _stagesMap;
            return Visit(body).Select(newbody => IR.F.Tensors.Boxing(newbody, body.CheckedType)).ToArray();
        }

        protected override IReadOnlyList<Expr> DefaultVisitLeaf(Expr expr)
        {
            return new[] { expr };
        }

        protected override IReadOnlyList<Expr> VisitLeafTuple(Tuple expr)
        {
            return expr.Fields.ToArray().Select(Visit).CartesianProduct().Select(e => new IR.Tuple(e.ToArray())).ToArray();
        }

        protected override IReadOnlyList<Expr> VisitLeafVar(Var expr)
        {
            return Array.Empty<Expr>();
        }

        protected override IReadOnlyList<Expr> VisitLeafConst(Const expr)
        {
            return Array.Empty<Expr>();
        }

        protected override IReadOnlyList<Expr> VisitLeafCall(Call expr)
        {
            if (expr.Target is not Op op)
            {
                throw new NotSupportedException("not support auto distributed call function");
            }

            var equivalArgs = op.Parameters.
                Select(param => GetLeafArgCandidates(param.ParameterKind, expr.Arguments[param.Index])).ToArray();
            var equivalCalls = equivalArgs.
                CartesianProduct().
                Select(args => args.ToArray()).
                Select(args => BuildEqualityCalls(op, args)).
                SelectMany(i => i).
                ToArray();

            if (equivalCalls.Any(t => t.Valid))
            {
                return equivalCalls.Where(t => t.Valid).Select(t => t.Call).ToArray();
            }

            var boxingArgs = new List<IReadOnlyList<Expr>>();
            foreach (var (info, newArgs) in op.Parameters.Zip(equivalArgs))
            {
                var oldArg = expr.Arguments[info.Index];
                if (!_stagesMap.TryGetValue(oldArg, out var equivals))
                {
                    equivals = new List<Expr>();
                    _stagesMap.Add(oldArg, equivals);
                }

                equivals.AddRange(newArgs);
                var tensorType = (TensorType)oldArg.CheckedType;
                boxingArgs.Add(info.ParameterKind switch
                {
                    ParameterKind.Input => DistributedUtility.GetLeafCandidateNDSBPs(tensorType, Placement).Select(ndsbp => IR.F.Tensors.Boxing(newArgs[0], new DistributedType(tensorType, ndsbp, Placement))).ToList(),
                    ParameterKind.Attribute => new Expr[] { newArgs[0] },
                    _ => throw new ArgumentOutOfRangeException(info.ParameterKind.ToString()),
                });
            }

            equivalCalls = boxingArgs.CartesianProduct().
                Select(Enumerable.ToArray<Expr>).
                Select(args => BuildEqualityCalls(op, args)).
                SelectMany(i => i).
                ToArray();

            if (!equivalCalls.Any(t => t.Valid))
            {
                throw new InvalidOperationException("after boxing still invalid!");
            }

            return equivalCalls.Where(t => t.Valid).Select(t => t.Call).ToArray();
        }

        private IReadOnlyList<Expr> GetLeafArgCandidates(ParameterKind parameterKind, Expr expr) => (parameterKind, expr) switch
        {
            (ParameterKind.Input, Expr e) when e is Const or Var => DistributedUtility.GetLeafCandidateBoxings(e, Placement),
            (ParameterKind.Input, Expr e) when e is IR.Tuple tp => tp.Fields.ToArray().Select(f => GetLeafArgCandidates(parameterKind, f)).CartesianProduct().Select(e => new IR.Tuple(e.ToArray())).ToArray(),
            (ParameterKind.Attribute, Expr e) when e is Const or Var => new[] { e },
            (_, Expr arg) => ExprMemo[arg],
        };

        private IEnumerable<(bool Valid, Call Call)> BuildEqualityCalls(Op target, Expr[] args)
        {
            if (!target.Parameters.Where(p => p.ParameterKind == ParameterKind.Input).All(p => IsDistributed(args[p.Index].CheckedType)))
            {
                throw new InvalidDataException();
            }

            var calls = new List<(bool, Call)>();
            var call = new Call(target, args);
            var valid = call.InferenceType();
            calls.Add((valid, call));
            if (!valid)
            {
                var broadcastArgs = args.Select(DistributedUtility.GetPartialCandidateBoxings).ToArray();

                if (!broadcastArgs.All(bargs => bargs.Count == 0))
                {
                    calls.AddRange(broadcastArgs.Select((bargs, i) => bargs.Any() ? bargs : bargs.Concat(new[] { args[i] })).
                        CartesianProduct().
                        Select(bargs => bargs.ToArray()).
                        Select(bargs => new Call(target, bargs)).
                        Select(c => (c.InferenceType(), c)));
                }
            }

            return calls;
        }

        private bool IsDistributed(IRType type) => type switch
        {
            DistributedType => true,
            TupleType t => t.All(IsDistributed),
            _ => false,
        };
    }

    private sealed class ConvertVisitor : ExprVisitor<Unit, Unit>
    {
        private readonly Dictionary<Expr, TIR.Buffer> _buffersMap = new(ReferenceEqualityComparer.Instance);
        private readonly List<Expr> _mainBody;

        public ConvertVisitor(List<Expr> mainBody)
        {
            _mainBody = mainBody;
        }

        public Fusion VisitRootFusion => (Fusion)VisitRoot!;

        public IEnumerable<TIR.Buffer> OutputBuffers => _buffersMap.Values.OfType<TIR.Buffer>().Where(b => b.MemSpan.Location.HasFlag(MemoryLocation.Output));

        public IEnumerable<TIR.Buffer> InputBuffers => _buffersMap.Values.OfType<TIR.Buffer>().Where(b => b.MemSpan.Location.HasFlag(MemoryLocation.Input));

        protected override Unit DefaultVisitLeaf(Expr expr)
        {
            return default;
        }

        protected override Unit VisitLeafCall(Call expr)
        {
            var arguments = expr.Arguments.AsValueEnumerable().Select(AllocOrGetBuffer).ToArray();
            var ret = AllocOrGetBuffer(expr);
            switch (expr.Target)
            {
                case CPUKernelOp kernelOp:
                    switch (kernelOp.Target)
                    {
                        case Unary unary:
                            GenerateUnary(unary, arguments, ret);
                            break;
                        case Binary binary:
                            GenerateBinary(binary, arguments, ret);
                            break;
                        case MatMul matmul:
                            GenerateMatmul(matmul, arguments, ret);
                            break;
                        case LayerNorm layernorm:
                            GenerateLayerNorm(layernorm, arguments, ret, (DistributedType)expr.Arguments[0].CheckedType);
                            break;
                        case Gather gather:
                            GenerateGather(gather, arguments, ret);
                        case Concat concat:
                            GenerateConcat(concat, ((IR.Tuple)expr.Arguments[0]).Fields.AsValueEnumerable().Select(AllocOrGetBuffer).ToArray(), ret);
                            break;
                        case Slice slice:
                            GenerateSlice(slice, arguments[0], ret, expr.Arguments[1], expr.Arguments[2], expr.Arguments[3], (DistributedType)expr.CheckedType);
                            break;
                        default:
                            throw new NotSupportedException();
                    }

                    break;
                case Boxing boxing:
                    GenerateBoxing(boxing, arguments, ret, expr);
                    break;
                default:
                    throw new NotSupportedException();
            }

            return default;
        }

        private void GenerateConcat(Concat concat, Buffer[] inputs, Buffer ret)
        {
            _mainBody.Add(IR.F.XPU.Concat(concat.Axis, inputs, ret));
        }

        private void GenerateSlice(Slice slice, Buffer input, Buffer output, Expr begins, Expr ends, Expr axes, DistributedType distributedType)
        {
            _mainBody.Add(IR.F.XPU.Slice(input, output, begins, ends, axes, distributedType));
        }

        private void GenerateBoxing(Boxing boxing, Buffer[] arguments, Buffer ret, Call expr)
        {
            switch (expr.Arguments[0].CheckedType, boxing.NewType)
            {
                case (TensorType tensorType, DistributedType distTensorType):
                    {
                        _mainBody.Add(IR.F.XPU.TDMALoad(ret, arguments[0], distTensorType.NdSBP, distTensorType.Placement));
                    }

                    break;
                case (DistributedType distTensorType, TensorType tensorType):
                    {
                        _mainBody.Add(IR.F.XPU.TDMAStore(arguments[0], ret, distTensorType.NdSBP, distTensorType.Placement));
                    }

                    break;
                default:
                    throw new NotSupportedException();
            }
        }

        private void GenerateUnary(IR.Math.Unary unary, ReadOnlySpan<Buffer> arguments, Buffer ret)
        {
            var input = arguments[IR.Math.Unary.Input.Index];
            _mainBody.Add(IR.F.XPU.Unary(unary.UnaryOp, input, ret));
        }

        private void GenerateBinary(Binary binary, Buffer[] arguments, Buffer ret)
        {
            _mainBody.Add(IR.F.XPU.Binary(binary.BinaryOp, arguments[0], arguments[1], ret));
        }

        private void GenerateMatmul(MatMul matmul, Buffer[] arguments, Buffer ret)
        {
            _mainBody.Add(IR.F.XPU.Matmul(arguments[0], arguments[1], ret));
        }

        private void GenerateLayerNorm(LayerNorm layerNorm, Buffer[] arguments, Buffer ret, DistributedType distributedType)
        {
            _mainBody.Add(IR.F.XPU.LayerNorm(layerNorm.Axis, layerNorm.Epsilon, layerNorm.UseMean, arguments[0], arguments[1], arguments[2], ret, distributedType));
        }

        private void GenerateGather(Gather gahter, Buffer[] arguments, Buffer ret)
        {
            _mainBody.Add(IR.F.XPU.Gather(gahter.Axis, arguments[0], arguments[1], ret));
        }

#if false
        private void GenerateUnary(IR.Math.Unary unary, ReadOnlySpan<Buffer> arguments, Buffer ret)
        {
            var loops = Enumerable.Range(0, input.Rank).Select(i => (T.ForLoop(out var loopVar, (0, input.Dimensions[i]), i == 0 ? LoopMode.Parallel : LoopMode.Serial, $"loop_{i}"), loopVar)).ToArray();
            var loopVars = loops.Select(f => f.loopVar).ToArray();
            Expr stmt = T.BufferStore(ret, loopVars, IR.F.Math.Unary(unary.UnaryOp, T.BufferLoad(input, loopVars)));
            var final = loops.Reverse().Aggregate(stmt, (acc, p) => p.Item1.Body(acc).Build());
            _mainBody.Add(T.Block(nameof(Unary)).Body(
                T.MatchBuffer(arguments[0]),
                T.MatchBuffer(ret),
                final).Build());
        }

        private void GenerateMatMul(ReadOnlySpan<Buffer> arguments, Buffer ret, Call expr)
        {
            var lhs = arguments[0];
            var rhs = arguments[1];
            var loops = Enumerable.Range(0, lhs.Rank - 2).Select(i => (T.ForLoop(out var loopVar, (0, lhs.Dimensions[i]), LoopMode.Serial, $"loop_{i}"), loopVar)).ToArray();
            var loopVars = loops.Select(f => f.loopVar).ToArray();
            var stmt = T.ForLoop(out var m, (0, lhs.Dimensions[^2]), LoopMode.Parallel).Body(
                T.Serial(out var n, (0, rhs.Dimensions[^1])).Body(
                    T.BufferStore(ret, loopVars.Concat(new[] { m, n }).ToArray(), 0f),
                    T.Serial(out var k, (0, lhs.Dimensions[^1])).Body(
                        T.BufferStore(ret, loopVars.Concat(new[] { m, n }).ToArray(), T.BufferLoad(ret, loopVars.Concat(new[] { m, n }).ToArray()) + (T.BufferLoad(lhs, loopVars.Concat(new[] { m, k }).ToArray()) * T.BufferLoad(rhs, loopVars.Concat(new[] { k, n }).ToArray())))))).
                Build();
            var final = loops.Reverse().Aggregate(stmt, (acc, p) => p.Item1.Body(acc).Build());
            // [m,k] @ [k, n]
            var body = T.Block(nameof(MatMul)).Body(
                T.MatchBuffer(arguments[0]),
                T.MatchBuffer(arguments[1]),
                T.MatchBuffer(ret),
                final);
            _mainBody.Add(body.Build());
        }

        private void GenerateBinary(Binary binary, ReadOnlySpan<Buffer> arguments, Buffer ret, Call call)
        {
            var lhs = call.Arguments[Binary.Lhs.Index];
            var rhs = call.Arguments[Binary.Rhs.Index];
            var lhsBuffer = arguments[Binary.Lhs.Index];
            var rhsBuffer = arguments[Binary.Rhs.Index];

            var outShape = call.CheckedShape.ToValueArray();
            var lhsShape = Enumerable.Repeat(1, outShape.Length).ToArray();
            Array.Copy(lhs.CheckedShape.ToValueArray(), 0, lhsShape, lhsShape.Length - lhs.CheckedShape.Rank, lhs.CheckedShape.Rank);
            var rhsShape = Enumerable.Repeat(1, outShape.Length).ToArray();
            Array.Copy(rhs.CheckedShape.ToValueArray(), 0, rhsShape, rhsShape.Length - rhs.CheckedShape.Rank, rhs.CheckedShape.Rank);

            var lhsScale = outShape.Zip(lhsShape).Select(s => s.First / s.Second).ToArray();
            var rhsScale = outShape.Zip(rhsShape).Select(s => s.First / s.Second).ToArray();

            var loops = Enumerable.Range(0, outShape.Length).Select(i => (T.ForLoop(out var loopVar, (0, outShape[i]), LoopMode.Serial, $"loop_{i}"), loopVar)).ToArray();
            var loopVars = loops.Select(f => f.loopVar).ToArray();
            var lhsLoopVars = loopVars.Zip(lhsScale).Select(v => v.First / v.Second).ToArray();
            var rhsLoopVars = loopVars.Zip(rhsScale).Select(v => v.First / v.Second).ToArray();
            Expr stmt = T.BufferStore(ret, loopVars, IR.F.Math.Binary(binary.BinaryOp, T.BufferLoad(lhsBuffer, lhsLoopVars), T.BufferLoad(rhsBuffer, rhsLoopVars)));
            var final = loops.Reverse().Aggregate(stmt, (acc, p) => p.Item1.Body(acc).Build());
            var body = T.Block(nameof(Binary)).Body(
                T.MatchBuffer(arguments[0]),
                T.MatchBuffer(arguments[1]),
                T.MatchBuffer(ret),
                final);
            _mainBody.Add(body.Build());
        }
#endif

        private TIR.Buffer AllocOrGetBuffer(Expr expr)
        {
            var name = $"buffer_{_buffersMap.Keys.Count}";
            if (!_buffersMap.TryGetValue(expr, out var buffer))
            {
                switch (expr)
                {
                    case Call c:
                        var (type, loc) = GetTypeAndLocation(c.CheckedType);
                        if (ReferenceEquals(c, VisitRootFusion.Body))
                        {
                            loc = MemoryLocation.Output;
                        }

                        buffer = T.AttachBuffer(type, loc, out _, out _, name);
                        break;
                    case Var v:
                        buffer = T.AttachBuffer((TensorType)v.CheckedType, MemoryLocation.Input, out _, out _, name);
                        break;
                    case TensorConst c:
                        buffer = T.AttachBuffer(c, out _, name);
                        break;
                    case IR.Tuple:
                        return null!;
                    default:
                        throw new NotSupportedException();
                }

                _buffersMap.Add(expr, buffer);
            }

            return buffer;
        }

        private Tuple<TensorType, MemoryLocation> GetTypeAndLocation(IRType type)
        {
            MemoryLocation location = MemoryLocation.Data;
            if (type is DistributedType distTensorType)
            {
                if (distTensorType.Placement.Rank == 2)
                {
                    location = MemoryLocation.L1Data;
                }
                else if (distTensorType.Placement.Rank == 1)
                {
                    location = MemoryLocation.L2Data;
                }
            }

            TensorType tensorType;
            if (type is DistributedType distTensor)
            {
                if (!DistributedUtility.IsDistributable(distTensor.TensorType, distTensor.NdSBP.ToArray(), distTensor.Placement, out var tType))
                {
                    throw new NotSupportedException();
                }
                else
                {
                    tensorType = tType;
                }
            }
            else if (type is TensorType ttype)
            {
                tensorType = ttype;
            }
            else
            {
                throw new NotSupportedException();
            }

            return new Tuple<TensorType, MemoryLocation>(tensorType, location);
        }
    }
}
