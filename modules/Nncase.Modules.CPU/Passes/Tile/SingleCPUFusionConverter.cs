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
        var candidatebodys = distConverter.Convert(fusion.Body);
        var graph = new EGraph();
        var bodyEclasses = candidatebodys.Select(graph.Add).ToArray();
        _ = bodyEclasses.Skip(1).Select(cls => graph.Union(bodyEclasses[0], cls)).ToArray(); // must keep this.
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
        public DistributeConvertVisitor(TileOptions tileOptions)
        {
            TileOptions = tileOptions;
            Placement = new Placement(Placement.DeviceKind.CPU, tileOptions.Hierarchy, "bt");
        }

        public TileOptions TileOptions { get; }

        public Placement Placement { get; }

        public IReadOnlyList<Expr> Convert(Expr body)
        {
            return Visit(body).Select(newbody => IR.F.Tensors.Boxing(newbody, body.CheckedType)).ToArray();
        }

        protected override IReadOnlyList<Expr> DefaultVisitLeaf(Expr expr)
        {
            return new[] { expr };
        }

        protected override IReadOnlyList<Expr> VisitLeafVar(Var expr)
        {
            var type = (TensorType)expr.TypeAnnotation;
            return DistributedUtilities.GetLeafCandidateNDSBPs(type, Placement).
                Select(ndsbp => IR.F.Tensors.Boxing(expr, new DistributedType(type, ndsbp, Placement))).
                ToArray();
        }

        protected override IReadOnlyList<Expr> VisitLeafConst(Const expr)
        {
            return DistributedUtilities.GetLeafCandidateNDSBPs((TensorType)expr.CheckedType, Placement)
                .Select(ndsbp => IR.F.Tensors.Boxing(expr, new DistributedType((TensorType)expr.CheckedType, ndsbp, Placement))).
                ToArray();
        }

        protected override IReadOnlyList<Expr> VisitLeafCall(Call expr)
        {
            return expr.Arguments.ToArray().
                Select(arg => ExprMemo[arg]).
                CartesianProduct<Expr>().
                Select(args => args.ToArray()).
                Select(args => new Call(expr.Target, args)).
                Where(c => c.InferenceType()).
                ToArray();
        }
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
            var arguments = expr.Arguments.AsValueEnumerable().Select(TryAllocateBuffer).ToArray();
            var ret = TryAllocateBuffer(expr);
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

        private void GenerateBoxing(Boxing boxing, Buffer[] arguments, Buffer ret, Call expr)
        {
            switch (expr.Arguments[0].CheckedType, boxing.NewType)
            {
                case (TensorType tensorType, DistributedType distTensorType):
                    {
                        _mainBody.Add(T.Block(nameof(Boxing)).Body(IR.F.XPU.TDMALoad(ret, arguments[0], distTensorType.NdSbp, distTensorType.Placement)).Build());
                    }

                    break;
                case (DistributedType distTensorType, TensorType tensorType):
                    {
                        _mainBody.Add(T.Block(nameof(Boxing)).Body(IR.F.XPU.TDMAStore(arguments[0], ret, distTensorType.NdSbp, distTensorType.Placement)).Build());
                    }

                    break;
                default:
                    throw new NotSupportedException();
            }
        }

        private void GenerateUnary(IR.Math.Unary unary, ReadOnlySpan<Buffer> arguments, Buffer ret)
        {
            var input = arguments[IR.Math.Unary.Input.Index];
            _mainBody.Add(T.Block(nameof(IR.Math.Unary)).Body(IR.F.XPU.Unary(unary.UnaryOp, input, ret)).Build());
        }

        private void GenerateBinary(Binary binary, Buffer[] arguments, Buffer ret)
        {
            _mainBody.Add(T.Block(nameof(IR.Math.Unary)).Body(IR.F.XPU.Binary(binary.BinaryOp, arguments[0], arguments[1], ret)).Build());
        }

        private void GenerateMatmul(MatMul matmul, Buffer[] arguments, Buffer ret)
        {
            if (ret.MemSpan.Location == MemoryLocation.L2Data)
            {
                _mainBody.Add(T.Block(nameof(XPU.BlockMMA)).Body(IR.F.XPU.Matmul(arguments[0], arguments[1], ret)).Build());
            }
            else
            {
                _mainBody.Add(T.Block(nameof(XPU.Matmul)).Body(IR.F.XPU.Matmul(arguments[0], arguments[1], ret)).Build());
            }
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

        private TIR.Buffer TryAllocateBuffer(Expr expr)
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
                    default:
                        throw new NotSupportedException();
                }

                _buffersMap.Add(expr, buffer);
            }

            return buffer;
        }

        private (TensorType, MemoryLocation) GetTypeAndLocation(IRType type)
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
                if (!DistributedUtilities.IsDistributable(distTensor.TensorType, distTensor.NdSbp.ToArray(), distTensor.Placement, out tensorType))
                {
                    throw new NotSupportedException();
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

            return (tensorType, location);
        }
    }
}
