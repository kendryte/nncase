// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Reactive;
using NetFabric.Hyperlinq;
using Nncase.IR;
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

    public TIRConvertVisitor(List<Expr> mainBody)
    {
        _mainBody = mainBody;
    }

    public Fusion VisitRootFusion => (Fusion)VisitRoot!;

    public IEnumerable<TIR.Buffer> OutputBuffers => _buffersMap.Values.OfType<TIR.Buffer>().Where(b => b.MemSpan.Location.HasFlag(MemoryLocation.Output));

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
                GenerateUnary(unary.UnaryOp.ToString().ToLower(System.Globalization.CultureInfo.CurrentCulture), arguments, ret);
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
                GenerateUnary("swish", arguments, ret);
                break;
            case IR.CPU.Boxing boxing:
                GenerateBoxing(boxing, arguments, ret, expr);
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
            default:
                throw new NotSupportedException();
        }

        return default;
    }

    private void GenerateReshape(Buffer input, Buffer ret)
    {
        _mainBody.Add(IR.F.XPU.ReShape(input, ret));
    }

    private void GenerateConcat(Concat concat, Buffer[] inputs, Buffer ret)
    {
        _mainBody.Add(IR.F.XPU.Concat(concat.Axis, inputs, ret));
    }

    private void GenerateSlice(Slice slice, Buffer input, Buffer output, Expr begins, Expr ends, Expr axes, DistributedType distributedType)
    {
        _mainBody.Add(IR.F.XPU.Slice(input, output, begins, ends, axes, distributedType));
    }

    private void GenerateBoxing(IR.CPU.Boxing boxing, Buffer[] arguments, Buffer ret, Call expr)
    {
        switch (expr.Arguments[0].CheckedType, boxing.NewType)
        {
            case (TensorType, DistributedType distTensorType):
                {
                    _mainBody.Add(IR.F.XPU.TDMALoad(ret, arguments[0], distTensorType.NdSBP, distTensorType.Placement));
                }

                break;
            case (DistributedType distTensorType, TensorType):
                {
                    _mainBody.Add(IR.F.XPU.TDMAStore(arguments[0], ret, distTensorType.NdSBP, distTensorType.Placement));
                }

                break;
            case (DistributedType inType, DistributedType outType):
                {
                    var partialSumPos = Enumerable.Range(0, inType.NdSBP.Count).Where(i => inType.NdSBP[i] is SBPPartialSum).Select(i => (i, outType.NdSBP[i])).ToArray();
                    if (partialSumPos.Length > 0)
                    {
                        var placement = inType.Placement with
                        {
                            Hierarchy = new IRArray<int>(partialSumPos.Select(t => inType.Placement.Hierarchy[t.i])),
                            Name = new string(partialSumPos.Select(t => inType.Placement.Name[t.i]).ToArray()),
                        };
                        _mainBody.Add(IR.F.XPU.GatherReduceScatter(arguments[0], ret, partialSumPos, placement));
                    }
                    else
                    {
                        _mainBody.Add(IR.F.XPU.TDMAStore(arguments[0], None.Default, inType.NdSBP, inType.Placement));
                        _mainBody.Add(IR.F.XPU.TDMALoad(ret, None.Default, outType.NdSBP, outType.Placement));
                    }
                }

                break;
            default:
                throw new NotSupportedException();
        }
    }

    private void GenerateUnary(string unaryOp, ReadOnlySpan<Buffer> arguments, Buffer ret)
    {
        var input = arguments[IR.Math.Unary.Input.Index];
        _mainBody.Add(IR.F.XPU.Unary(unaryOp, input, ret));
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

    private void GenerateInstanceNorm(InstanceNormalization instNorm, float eps, Buffer[] arguments, Buffer ret, DistributedType distributedType)
    {
        _mainBody.Add(IR.F.XPU.InstanceNorm(eps, arguments[0], arguments[1], arguments[2], ret, distributedType));
    }

    private void GenerateGather(Gather gahter, Buffer[] arguments, Buffer ret)
    {
        _mainBody.Add(IR.F.XPU.Gather(gahter.Axis, arguments[0], arguments[1], ret));
    }

    private void GenerateSoftmax(Softmax softmax, int axis, Buffer[] arguments, Buffer ret, DistributedType distributedType)
    {
        _mainBody.Add(IR.F.XPU.Softmax(axis, arguments[0], ret, distributedType));
    }

    private void GenerateTranspose(Transpose transpose, int[] perm, Buffer[] arguments, Buffer ret)
    {
        _mainBody.Add(IR.F.XPU.Transpose(perm, arguments[0], ret));
    }

    private void GenerateConv2D(Conv2D conv, Buffer[] arguments, Buffer ret, int[] stride, int[] padding, int[] dilation, int groups, TensorConst fusedClamp, DistributedType distributedType)
    {
        _mainBody.Add(IR.F.XPU.Conv2D(arguments[0], arguments[1], arguments[2], ret, stride, padding, dilation, groups, fusedClamp, distributedType));
    }

    private void GenerateReduceArg(ReduceArg reduceArg, Buffer[] arguments, Buffer ret, int axis, bool keepdims, bool selectLastIndex, ReduceArgOp op, DataType dataType)
    {
        _mainBody.Add(IR.F.XPU.ReduceArg(arguments[0], ret, axis, keepdims, selectLastIndex, op, dataType));
    }

    private void GenerateResize(ResizeImage resize, Buffer[] arguments, Buffer ret, float[] roi, int[] newSize, float cubicCoeffA, int excludeOutside, float extrapolationValue, DistributedType distributedType)
    {
        _mainBody.Add(IR.F.XPU.Resize(arguments[0], ret, roi, newSize, cubicCoeffA, excludeOutside, extrapolationValue, resize.ResizeMode, resize.TransformationMode, resize.NearestMode, resize.IsTFResize));
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
                case IR.None:
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
