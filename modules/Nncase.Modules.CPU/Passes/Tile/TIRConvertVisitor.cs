// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Reactive;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.CPU;
using Nncase.IR.Imaging;
using Nncase.IR.Math;
using Nncase.IR.NN;
using Nncase.IR.Tensors;
using Nncase.TIR;
using Nncase.Utilities;
using Buffer = Nncase.TIR.Buffer;

namespace Nncase.Passes.Tile;

public sealed class TIRConvertVisitor : ExprVisitor<Unit, Unit>
{
    private readonly Dictionary<Expr, TIR.Buffer> _buffersMap = new(ReferenceEqualityComparer.Instance);
    private readonly List<Expr> _mainBody;
    private readonly List<(int, TIR.Buffer)> _outputbuffers;

    public TIRConvertVisitor(List<Expr> mainBody)
    {
        _mainBody = mainBody;
        _outputbuffers = new();
    }

    public Fusion VisitRootFusion => (Fusion)VisitRoot!;

    public IEnumerable<TIR.Buffer> OutputBuffers => _outputbuffers.OrderBy(p => p.Item1).Select(p => p.Item2);

    public IEnumerable<TIR.Buffer> InputBuffers => VisitRootFusion.Parameters.ToArray().Select(p => _buffersMap[p]).OfType<TIR.Buffer>().Where(b => b.MemSpan.Location.HasFlag(MemoryLocation.Input));

    protected override Unit DefaultVisitLeaf(Expr expr)
    {
        return default;
    }

    protected override Unit VisitLeafCall(Call expr)
    {
        var arguments = expr.Arguments.AsValueEnumerable().Select(AllocOrGetBuffer).ToArray();
        var ret = AllocOrGetBuffer(expr);
        var op = expr.Target is IR.CPU.CPUKernelOp kop ? kop.Target : expr.Target;
        switch (op)
        {
            case Unary unary:
                GenerateUnary(unary.UnaryOp, arguments, ret);
                break;
            case IR.CPU.Boxing boxing:
                GenerateBoxing(boxing, arguments, ret, expr);
                break;
#if false
            case Binary binary:
                GenerateBinary(binary, arguments, ret, expr);
                break;
            case MatMul matmul:
                GenerateMatmul(matmul, arguments, ret);
                break;
            case LayerNorm layernorm:
                GenerateLayerNorm(layernorm, arguments, ret, (DistributedType)expr.Arguments[0].CheckedType);
                break;
            case InstanceNormalization instnorm:
                GenerateInstanceNorm(instnorm, ((TensorConst)expr.Arguments[3]).Value.ToScalar<float>(), arguments, ret, (DistributedType)expr.Arguments[0].CheckedType);
                break;
            case Gather gather:
                GenerateGather(gather, arguments, ret);
                break;
            case Concat concat:
                GenerateConcat(concat, ((IR.Tuple)expr.Arguments[0]).Fields.AsValueEnumerable().Select(AllocOrGetBuffer).ToArray(), ret);
                break;
            case Slice slice:
                GenerateSlice(slice, arguments[0], ret, expr.Arguments[1], expr.Arguments[2], expr.Arguments[3], (DistributedType)expr.CheckedType);
                break;
            case Softmax softmax:
                GenerateSoftmax(softmax, ((TensorConst)expr.Arguments[1]).Value.ToScalar<int>(), arguments, ret, (DistributedType)expr.CheckedType);
                break;
            case Transpose transpose:
                GenerateTranspose(transpose, ((TensorConst)expr.Arguments[1]).Value.ToArray<int>(), arguments, ret);
                break;
            case Reshape or Unsqueeze:
                GenerateReshape(arguments[0], ret);
                break;
            case Swish:
                GenerateSwishB(arguments[0], ret, ((TensorConst)expr.Arguments[1]).Value.ToScalar<float>());
                break;
            case Gelu:
                GenerateUnary("gelu", arguments, ret);
                break;
            case Conv2D conv:
                GenerateConv2D(conv, arguments, ret, ((TensorConst)expr.Arguments[3]).Value.ToArray<int>(), ((TensorConst)expr.Arguments[4]).Value.ToArray<int>(), ((TensorConst)expr.Arguments[5]).Value.ToArray<int>(), ((TensorConst)expr.Arguments[6]).Value.ToScalar<int>(), (TensorConst)expr.Arguments[7], (DistributedType)expr.CheckedType);
                break;
            case ReduceArg reduceArg:
                GenerateReduceArg(reduceArg, arguments, ret, ((TensorConst)expr.Arguments[1]).Value.ToScalar<int>(), ((TensorConst)expr.Arguments[2]).Value.ToScalar<bool>(), ((TensorConst)expr.Arguments[3]).Value.ToScalar<bool>(), reduceArg.ReduceArgOp, reduceArg.DestType);
                break;
            case ResizeImage resize:
                float[] roi = expr.Arguments[1] is TensorConst tc ? tc.Value.ToArray<float>() : new[] { 0f, 0f, 1f, 1f };
                int[] newSize = ((TensorConst)expr.Arguments[2]).Value.ToArray<int>();
                float cubicCoeffA = expr.Arguments[3] is TensorConst tc1 ? tc1.Value.ToScalar<float>() : -0.75f;
                int excludeOutside = expr.Arguments[4] is TensorConst tc2 ? tc2.Value.ToScalar<int>() : 0;
                float extrapolationValue = expr.Arguments[5] is TensorConst tc3 ? tc3.Value.ToScalar<float>() : 0f;
                GenerateResize(resize, arguments, ret, roi, newSize, cubicCoeffA, excludeOutside, extrapolationValue, (DistributedType)expr.CheckedType);
                break;
            case Cast cast:
                GenerateCast(cast.NewType, cast.CastMode, arguments, ret);
                break;
            case Expand expand:
                GenerateExpand(((TensorConst)expr.Arguments[1]).Value.ToArray<int>(), (DistributedType)expr.CheckedType, arguments, ret);
                break;
            case Clamp clamp:
                GenerateClamp(arguments, ret, ((TensorConst)expr.Arguments[1]).Value.ToArray<float>()[0], ((TensorConst)expr.Arguments[2]).Value.ToArray<float>()[0]);
                break;
#endif
            default:
                throw new NotSupportedException();
        }

        return default;
    }

    private void GenerateUnary(UnaryOp unaryOp, ReadOnlySpan<Buffer> arguments, Buffer ret)
    {
        var input = arguments[IR.Math.Unary.Input.Index];
        _mainBody.Add(TIR.F.CPU.Unary(unaryOp, input, ret));
    }

    private void GenerateBoxing(IR.CPU.Boxing boxing, Buffer[] arguments, Buffer ret, Call expr)
    {
        switch (expr.Arguments[0].CheckedType, boxing.NewType)
        {
            case (TensorType, DistributedType distTensorType):
                {
                    _mainBody.Add(TIR.F.CPU.TensorLoad(ret, arguments[0], distTensorType.NdSBP, distTensorType.Placement));
                }

                break;
            case (DistributedType distTensorType, TensorType):
                {
                    _mainBody.Add(TIR.F.CPU.TensorStore(arguments[0], ret, distTensorType.NdSBP, distTensorType.Placement));
                }

                break;
            case (DistributedType inType, DistributedType outType):
                {
                    if (inType.NdSBP.Any(sbp => sbp is SBPPartialSum))
                    {
                        // _mainBody.Add(TIR.F.CPU.GatherReduceScatter(arguments[0], ret, inType, outType));
                    }
                    else
                    {
                        _mainBody.Add(TIR.F.CPU.TensorStore(arguments[0], None.Default, inType.NdSBP, inType.Placement));
                        _mainBody.Add(TIR.F.CPU.TensorLoad(ret, None.Default, outType.NdSBP, outType.Placement));
                    }
                }

                break;
            default:
                throw new NotSupportedException();
        }
    }

#if false
    private void GenerateSwishB(Buffer input, Buffer ret, float beta)
    {
        _mainBody.Add(TIR.F.CPU.SwishB(input, ret, beta));
    }

    private void GenerateReshape(Buffer input, Buffer ret)
    {
        _mainBody.Add(TIR.F.CPU.ReShape(input, ret));
    }

    private void GenerateConcat(Concat concat, Buffer[] inputs, Buffer ret)
    {
        _mainBody.Add(TIR.F.CPU.Concat(concat.Axis, inputs, ret));
    }

    private void GenerateSlice(Slice slice, Buffer input, Buffer output, Expr begins, Expr ends, Expr axes, DistributedType distributedType)
    {
        _mainBody.Add(TIR.F.CPU.Slice(input, output, begins, ends, axes, distributedType));
    }

    private void GenerateBoxing(IR.CPU.Boxing boxing, Buffer[] arguments, Buffer ret, Call expr)
    {
        switch (expr.Arguments[0].CheckedType, boxing.NewType)
        {
            case (TensorType, DistributedType distTensorType):
                {
                    _mainBody.Add(TIR.F.CPU.TDMALoad(ret, arguments[0], distTensorType.NdSBP, distTensorType.Placement));
                }

                break;
            case (DistributedType distTensorType, TensorType):
                {
                    _mainBody.Add(TIR.F.CPU.TDMAStore(arguments[0], ret, distTensorType.NdSBP, distTensorType.Placement));
                }

                break;
            case (DistributedType inType, DistributedType outType):
                {
                    if (inType.NdSBP.Any(sbp => sbp is SBPPartialSum))
                    {
                        _mainBody.Add(TIR.F.CPU.GatherReduceScatter(arguments[0], ret, inType, outType));
                    }
                    else
                    {
                        _mainBody.Add(TIR.F.CPU.TDMAStore(arguments[0], None.Default, inType.NdSBP, inType.Placement));
                        _mainBody.Add(TIR.F.CPU.TDMALoad(ret, None.Default, outType.NdSBP, outType.Placement));
                    }
                }

                break;
            default:
                throw new NotSupportedException();
        }
    }

    private void GenerateBinary(Binary binary, Buffer[] arguments, Buffer ret, Call expr)
    {
        var lhs = (DistributedType)expr.Arguments[0].CheckedType;
        var rhs = (DistributedType)expr.Arguments[1].CheckedType;
        var outtype = (DistributedType)expr.CheckedType;
        _mainBody.Add(TIR.F.CPU.Binary(binary.BinaryOp, lhs, rhs, outtype, arguments[0], arguments[1], ret));
    }

    private void GenerateMatmul(MatMul matmul, Buffer[] arguments, Buffer ret)
    {
        _mainBody.Add(TIR.F.CPU.Matmul(arguments[0], arguments[1], ret));
    }

    private void GenerateLayerNorm(LayerNorm layerNorm, Buffer[] arguments, Buffer ret, DistributedType distributedType)
    {
        _mainBody.Add(TIR.F.CPU.LayerNorm(layerNorm.Axis, layerNorm.Epsilon, layerNorm.UseMean, arguments[0], arguments[1], arguments[2], ret, distributedType));
    }

    private void GenerateInstanceNorm(InstanceNormalization instNorm, float eps, Buffer[] arguments, Buffer ret, DistributedType distributedType)
    {
        _mainBody.Add(TIR.F.CPU.InstanceNorm(eps, arguments[0], arguments[1], arguments[2], ret, distributedType));
    }

    private void GenerateGather(Gather gahter, Buffer[] arguments, Buffer ret)
    {
        _mainBody.Add(TIR.F.CPU.Gather(gahter.Axis, arguments[0], arguments[1], ret));
    }

    private void GenerateSoftmax(Softmax softmax, int axis, Buffer[] arguments, Buffer ret, DistributedType distributedType)
    {
        _mainBody.Add(TIR.F.CPU.Softmax(axis, arguments[0], ret, distributedType));
    }

    private void GenerateTranspose(Transpose transpose, int[] perm, Buffer[] arguments, Buffer ret)
    {
        _mainBody.Add(TIR.F.CPU.Transpose(perm, arguments[0], ret));
    }

    private void GenerateConv2D(Conv2D conv, Buffer[] arguments, Buffer ret, int[] stride, int[] padding, int[] dilation, int groups, TensorConst fusedClamp, DistributedType distributedType)
    {
        _mainBody.Add(TIR.F.CPU.Conv2D(arguments[0], arguments[1], arguments[2], ret, stride, padding, dilation, groups, fusedClamp, distributedType));
    }

    private void GenerateReduceArg(ReduceArg reduceArg, Buffer[] arguments, Buffer ret, int axis, bool keepdims, bool selectLastIndex, ReduceArgOp op, DataType dataType)
    {
        _mainBody.Add(TIR.F.CPU.ReduceArg(arguments[0], ret, axis, keepdims, selectLastIndex, op, dataType));
    }

    private void GenerateResize(ResizeImage resize, Buffer[] arguments, Buffer ret, float[] roi, int[] newSize, float cubicCoeffA, int excludeOutside, float extrapolationValue, DistributedType distributedType)
    {
        _mainBody.Add(TIR.F.CPU.Resize(arguments[0], ret, roi, newSize, cubicCoeffA, excludeOutside, extrapolationValue, resize.ResizeMode, resize.TransformationMode, resize.NearestMode, resize.IsTFResize));
    }

    private void GenerateCast(DataType dataType, CastMode castMode, ReadOnlySpan<Buffer> arguments, Buffer ret)
    {
        _mainBody.Add(TIR.F.CPU.Cast(arguments[0], ret, dataType, castMode));
    }

    private void GenerateExpand(int[] shape, DistributedType distributedType, ReadOnlySpan<Buffer> arguments, Buffer ret)
    {
        _mainBody.Add(TIR.F.CPU.Expand(shape, distributedType, arguments[0], ret));
    }

    private void GenerateClamp(ReadOnlySpan<Buffer> arguments, Buffer ret, float min, float max)
    {
        _mainBody.Add(TIR.F.CPU.Clamp(arguments[0], ret, min, max));
    }
#endif

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
                    {
                        var loc = GetMemLocation(c.CheckedType);
                        var index = CheckRootCall(c, ref loc);
                        if (c.Target is Boxing box && box.NewType is DistributedType d && !d.TensorType.Shape.Equals(c.Arguments[0].CheckedShape))
                        {
                            name += "_reshape";
                        }

                        TensorType? dividedType = null;
                        if (c.CheckedType is TensorType tensorType)
                        {
                            dividedType = tensorType;
                        }
                        else if (c.CheckedType is DistributedType distributedType)
                        {
                            if (DistributedUtility.TryGetDividedTensorType(distributedType, out var type))
                            {
                                dividedType = type;
                            }
                        }

                        if (dividedType is not null)
                        {
                            buffer = T.AttachBuffer(dividedType, loc, out _, out _, name);
                        }
                        else if (c.CheckedType is DistributedType distributedType)
                        {
                            var shape = DistributedUtility.TryGetNonUniformDividedShape(distributedType);
                            var @var = new Var(TensorType.Pointer(distributedType.TensorType.DType));
                            var strides = TensorUtilities.GetStrides(shape);
                            var size = TensorUtilities.GetProduct(shape) * distributedType.TensorType.DType.SizeInBytes;
                            buffer = new Buffer(name, distributedType.TensorType.DType, new MemSpan(@var, size, loc), shape, strides);
                        }
                        else
                        {
                            throw new NotSupportedException();
                        }

                        if (index != -1)
                        {
                            _outputbuffers.Add((index, buffer));
                        }
                    }

                    break;
                case Var v:
                    buffer = T.AttachBuffer((TensorType)v.CheckedType, MemoryLocation.Input, out _, out _, name);
                    break;
                case TensorConst c:
                    buffer = T.AttachBuffer(c, out _, name);
                    break;
                case IR.Tuple:
                    return null!;
                case IR.None:
                    return null!;
                default:
                    throw new NotSupportedException();
            }

            _buffersMap.Add(expr, buffer);
        }

        return buffer;
    }

    private int CheckRootCall(Expr c, ref MemoryLocation loc)
    {
        var index = -1;
        if (VisitRootFusion.Body is Call rootCall && ReferenceEquals(c, rootCall))
        {
            index = 0;
            loc = MemoryLocation.Output;
        }
        else if (VisitRootFusion.Body is IR.Tuple tp)
        {
            for (int i = 0; i < tp.Fields.Length; i++)
            {
                if (ReferenceEquals(tp.Fields[i], c))
                {
                    index = i;
                    loc = MemoryLocation.Output;
                }
            }
        }

        return index;
    }

    private MemoryLocation GetMemLocation(IRType type)
    {
        MemoryLocation location = MemoryLocation.Data;
        if (type is DistributedType)
        {
            location = MemoryLocation.L1Data;
        }

        return location;
    }
}
