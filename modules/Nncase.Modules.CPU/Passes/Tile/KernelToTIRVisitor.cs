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

public sealed class KernelToTIRVisitor : ExprVisitor<Unit, Unit>
{
    private readonly Dictionary<Expr, TIR.Buffer> _buffersMap = new(ReferenceEqualityComparer.Instance);
    private readonly List<Expr> _mainBody;
    private readonly HashSet<PrimFunction> _devices;
    private readonly List<(int, TIR.Buffer)> _outputbuffers;
    private readonly Dictionary<Fusion, FusionChecker> _fusionCheckCache;

    public KernelToTIRVisitor(List<Expr> mainBody, HashSet<PrimFunction> devices, Dictionary<Fusion, FusionChecker> fusionCheckCache)
    {
        _mainBody = mainBody;
        _devices = devices;
        _outputbuffers = new();
        _fusionCheckCache = fusionCheckCache;
        VisitRootFusion = null!;
        DataUsage = 0;
        MaxDTypeSize = 0;
    }

    public ulong DataUsage { get; private set; }

    public ulong MaxDTypeSize { get; private set; }

    public Fusion VisitRootFusion { get; private set; }

    public IEnumerable<TIR.Buffer> OutputBuffers => _outputbuffers.OrderBy(p => p.Item1).Select(p => p.Item2);

    public IEnumerable<TIR.Buffer> InputBuffers => VisitRootFusion.Parameters.ToArray().Select(p => _buffersMap[p]).OfType<TIR.Buffer>().Where(b => b.MemSpan.Location.HasFlag(MemoryLocation.Input));

    public void Convert(Fusion post)
    {
        VisitRootFusion = post;
        AllocBuffers(post);
        Visit(post);
    }

    protected override Unit DefaultVisitLeaf(Expr expr)
    {
        return default;
    }

    protected override Unit VisitLeafCall(Call expr)
    {
        var arguments = expr.Arguments.AsValueEnumerable().Select(GetBuffer).ToArray();
        var ret = GetBuffer(expr);
        var op = expr.Target;
        switch (op)
        {
            case PrimFunctionWrapper { Target: TIR.PrimFunction { ModuleKind: string mkind } deviceFunc }:
                _devices.Add(deviceFunc);
                _mainBody.Add(new Call(deviceFunc, arguments.Concat(new[] { ret }).ToArray()));
                break;
            case Fusion deviceFunc:
                {
                    var r = new DeviceFusionToPrimFuncRewriter(_fusionCheckCache);
                    var post = (TIR.PrimFunction)r.Rewrite(deviceFunc);
                    _devices.Add(post);
                    _mainBody.Add(new Call(post, arguments.Concat(new[] { ret }).ToArray()));
                }

                break;
            case IR.Math.Unary unary:
                GenerateUnary(unary.UnaryOp, arguments, ret);
                break;
            case IR.CPU.Boxing boxing:
                GenerateBoxing(boxing, arguments, ret, expr);
                break;
            case Binary binary:
                GenerateBinary(binary, arguments, ret, expr);
                break;
            case IR.CPU.Pack pack:
                _mainBody.Add(TIR.F.CPU.Pack(arguments[0], ret, pack.Lanes, pack.Axes));
                break;
            case IR.CPU.Unpack unpack:
                _mainBody.Add(TIR.F.CPU.Unpack(arguments[0], ret, unpack.Axes));
                break;
            case IR.CPU.PackedBinary packed_binary:
                // _mainBody.Add(TIR.F.CPU.Binary(arguments[0], arguments[1], ret, packed_binary.BinaryOp, packed_binary.LhsPackedAxes, packed_binary.LhsPadedNums, packed_binary.RhsPackedAxes, packed_binary.RhsPadedNums));
                _mainBody.Add(TIR.F.CPU.Binary(packed_binary.BinaryOp, arguments[0], arguments[1], ret));
                break;
            case IR.CPU.PackedMatMul packed_mat_mul:
                _mainBody.Add(TIR.F.CPU.PackedMatMul(arguments[0], arguments[1], ret, packed_mat_mul.LhsPackedAxes, packed_mat_mul.LhsPadedNums, packed_mat_mul.RhsPackedAxes, packed_mat_mul.RhsPadedNums));
                break;
            case IR.Math.MatMul matmul:
                _mainBody.Add(TIR.F.CPU.Matmul(arguments[0], arguments[1], ret, None.Default));
                break;
            case IR.NN.Conv2D conv:
                {
                    var input = expr[IR.NN.Conv2D.Input];
                    var weights = expr[IR.NN.Conv2D.Weights];
                    var bias = expr[IR.NN.Conv2D.Bias];
                    var strides = ((TensorConst)expr[IR.NN.Conv2D.Stride]).Value.ToArray<int>();
                    var padding = ((TensorConst)expr[IR.NN.Conv2D.Padding]).Value.ToArray<int>();
                    var dilation = ((TensorConst)expr[IR.NN.Conv2D.Dilation]).Value.ToArray<int>();
                    var groups = ((TensorConst)expr[IR.NN.Conv2D.Groups]).Value.ToScalar<int>();
                    var fusedClamp = ((TensorConst)expr[IR.NN.Conv2D.FusedClamp]).Value.ToArray<float>();
                    var wShape = weights.CheckedShape.ToValueArray();
                    var outShape = expr.CheckedShape.ToValueArray();
                    if (fusedClamp[0] != float.NegativeInfinity || fusedClamp[1] != float.PositiveInfinity || conv.PadMode != PadMode.Constant)
                    {
                        throw new NotSupportedException("not support this conv2d");
                    }

                    _mainBody.Add(TIR.F.CPU.Conv2D(arguments[0], arguments[1], arguments[2], ret, strides, padding, dilation, groups, conv.PadMode, (DistributedType)expr.CheckedType));
                }

                break;
            case IR.CPU.Im2col im2col:
                _mainBody.Add(TIR.F.CPU.Im2col(arguments[0], ret, im2col.Kernel, im2col.Stride, im2col.Padding, im2col.PackedAxes, im2col.PadedNums));
                break;
            case IR.CPU.PackedSoftmax packed_softmax:
                _mainBody.Add(TIR.F.CPU.PackedSoftmax(arguments[0], ret, packed_softmax.Axis, packed_softmax.PackedAxes));
                break;
            case IR.NN.Softmax softmax:
                _mainBody.Add(TIR.F.CPU.PackedSoftmax(arguments[0], ret, ((TensorConst)expr.Arguments[1]).Value.ToScalar<int>(), Array.Empty<int>()));
                break;
            case IR.CPU.PackedTranspose packed_transpose:
                // _mainBody.Add(TIR.F.CPU.PackedTranspose(arguments[0], arguments[1], ret, packed_transpose.PackedAxes));
                _mainBody.Add(TIR.F.CPU.PackedTranspose(arguments[0], ret, ((TensorConst)expr.Arguments[1]).Value.ToArray<int>(), packed_transpose.PackedAxes));
                break;
            case IR.CPU.PackedLayerNorm packed_layer_norm:
                _mainBody.Add(TIR.F.CPU.PackedLayerNorm(arguments[0], arguments[1], arguments[2], ret, packed_layer_norm.Axis, packed_layer_norm.Epsilon, packed_layer_norm.UseMean, packed_layer_norm.PackedAxes, packed_layer_norm.PadedNums));
                break;
            case IR.NN.LayerNorm layernorm:
                _mainBody.Add(TIR.F.CPU.PackedLayerNorm(arguments[0], arguments[1], arguments[2], ret, layernorm.Axis, layernorm.Epsilon, layernorm.UseMean, Array.Empty<int>(), Array.Empty<int>()));
                break;
            case IR.NN.InstanceNormalization instancenorm:
                _mainBody.Add(TIR.F.CPU.InstanceNorm(arguments[0], arguments[1], arguments[2], ret, ((TensorConst)expr.Arguments[3]).Value.ToScalar<int>(), Array.Empty<int>(), Array.Empty<int>(), (DistributedType)expr.CheckedType));
                break;
            case IR.CPU.InstacneNorm instancenorm:
                _mainBody.Add(TIR.F.CPU.InstanceNorm(arguments[0], arguments[1], arguments[2], ret, instancenorm.Epsilon, instancenorm.PackedAxes, instancenorm.PadedNums, (DistributedType)expr.CheckedType));
                break;
            case IR.Imaging.ResizeImage resize:
                if (expr.Arguments[1] is not None || resize.IsTFResize)
                {
                    throw new NotSupportedException("not support tf resize");
                }

                _mainBody.Add(TIR.F.CPU.ResizeImage(arguments[0], ret, Array.Empty<int>(), Array.Empty<int>(), ((TensorConst)expr.Arguments[2]).Value.ToArray<int>(), resize.ResizeMode, resize.TransformationMode, resize.NearestMode));
                break;
            case IR.CPU.ResizeImage resize:
                _mainBody.Add(TIR.F.CPU.ResizeImage(arguments[0], ret, resize.PackedAxes.ToArray(), resize.PadedNums.ToArray(), resize.NewSize.ToArray(), resize.ResizeMode, resize.TransformationMode, resize.NearestMode));
                break;
            case IR.Tensors.Unsqueeze unsqueeze:
                _mainBody.Add(TIR.F.CPU.Reshape(arguments[0], ret, expr.CheckedShape.ToValueArray()));
                break;
            case IR.Tensors.Reshape reshape:
                _mainBody.Add(TIR.F.CPU.Reshape(arguments[0], ret, expr.CheckedShape.ToValueArray()));
                break;
            case IR.Tensors.Slice slice:
                _mainBody.Add(TIR.F.CPU.Slice(arguments[0], ret, ((TensorConst)expr.Arguments[1]).Value.ToArray<int>(), ((TensorConst)expr.Arguments[2]).Value.ToArray<int>(), ((TensorConst)expr.Arguments[3]).Value.ToArray<int>(), ((TensorConst)expr.Arguments[4]).Value.ToArray<int>(), (DistributedType)expr.CheckedType));
                break;
            case IR.Tensors.Concat concat:
                _mainBody.Add(TIR.F.CPU.Concat(((IR.Tuple)expr.Arguments[0]).Fields.AsValueEnumerable().Select(GetBuffer).ToArray(), ret, concat.Axis));
                break;
            case IR.Tensors.Transpose trans:
                _mainBody.Add(TIR.F.CPU.Transpose(arguments[0], ret, ((TensorConst)expr.Arguments[1]).Value.ToArray<int>()));
                break;
            case IR.NN.Swish swish:
                _mainBody.Add(TIR.F.CPU.Swish(arguments[0], ret, ((TensorConst)expr.Arguments[1]).Value.ToScalar<float>()));
                break;
            case IR.Tensors.Gather gather:
                _mainBody.Add(TIR.F.CPU.Gather(arguments[0], arguments[1], ret, gather.Axis));
                break;
            case IR.NN.Pad pad:
                _mainBody.Add(TIR.F.CPU.Pad(arguments[0], ret, ((TensorConst)expr.Arguments[1]).Value.ToArray<int>(), ((TensorConst)expr.Arguments[2]).Value.ToArray<float>()[0]));
                break;
            case IR.Math.Reduce reduce:
                _mainBody.Add(TIR.F.CPU.Reduce(arguments[0], ret, Array.Empty<int>(), Array.Empty<int>(), ((TensorConst)expr.Arguments[1]).Value.ToArray<int>().OrderBy(a => a).ToArray(), ((TensorConst)expr.Arguments[3]).Value.ToArray<bool>()[0], reduce.ReduceOp));
                break;
            default:
                throw new NotSupportedException();
        }

        return default;
    }

    private TIR.Buffer GetBuffer(Expr expr) => _buffersMap.GetValueOrDefault(expr, null!);

    private void AllocBuffers(Fusion fusion)
    {
        var candidates = ExprCollector.Collect(fusion).Where(e => e is Call or Var or TensorConst);
        MaxDTypeSize = (ulong)candidates.Select(e => e.CheckedDataType.SizeInBytes).Max();
        foreach (var expr in candidates)
        {
            var name = $"buffer_{_buffersMap.Keys.Count}";
            if (!_buffersMap.TryGetValue(expr, out var buffer))
            {
                switch (expr)
                {
                    case Call c:
                        var loc = MemoryLocation.Data;
                        var hierarchy = 1;
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
                            hierarchy = 1;
                            if (DistributedUtility.TryGetDividedTensorType(distributedType, out var type))
                            {
                                dividedType = type;
                            }
                        }

                        if (dividedType is TensorType)
                        {
                            T.AttachBuffer(Tensor.FromPointer(DataUsage, dividedType.DType), dividedType, loc, hierarchy, out buffer, name);
                            DataUsage += Enumerable.Range(0, dividedType.Shape.Rank).Aggregate(1ul, (size, i) => size * (ulong)dividedType.Shape[i].FixedValue) * (ulong)dividedType.DType.SizeInBytes;
                            DataUsage = MathUtility.AlignUp(DataUsage, MaxDTypeSize);
                        }
                        else if (c.CheckedType is DistributedType)
                        {
                            // deal the not uinform sbp.
                            // var shape = DistributedUtility.TryGetNonUniformDividedShape(distributedType);
                            // var @var = new Var(TensorType.Pointer(distributedType.TensorType.DType));
                            // var strides = TensorUtilities.GetStrides(shape);
                            // var size = TensorUtilities.GetProduct(shape) * distributedType.TensorType.DType.SizeInBytes;
                            // buffer = new Buffer(name, distributedType.TensorType.DType, new MemSpan(@var, size, loc, hierarchy), shape, strides);
                            throw new NotSupportedException("not support non uniform sbp");
                        }
                        else
                        {
                            throw new NotSupportedException();
                        }

                        if (index != -1)
                        {
                            _outputbuffers.Add((index, buffer));
                        }

                        break;
                    case Var v:
                        buffer = T.AttachBuffer((TensorType)v.CheckedType, MemoryLocation.Input, 1, out _, out _, name);
                        break;
                    case TensorConst c:
                        buffer = T.AttachBuffer(c, out _, name);
                        break;
                    default:
                        throw new NotSupportedException();
                }

                _buffersMap.Add(expr, buffer);
            }
        }
    }

    private void GenerateUnary(UnaryOp unaryOp, ReadOnlySpan<Buffer> arguments, Buffer ret)
    {
        var input = arguments[IR.Math.Unary.Input.Index];
        _mainBody.Add(TIR.F.CPU.Unary(unaryOp, input, ret));
    }

    private void GenerateBinary(Binary binary, Buffer[] arguments, Buffer ret, Call expr)
    {
        _mainBody.Add(TIR.F.CPU.Binary(binary.BinaryOp, arguments[0], arguments[1], ret));
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
                        _mainBody.Add(TIR.F.CPU.GatherReduceScatter(arguments[0], ret, inType, outType));
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

    private void GenerateWhere(ReadOnlySpan<Buffer> arguments, Buffer ret, DistributedType distributedType)
    {
        _mainBody.Add(TIR.F.CPU.Where(arguments[0], arguments[1], arguments[2], ret, distributedType));
    }
#endif

    private int CheckRootCall(Call c, ref MemoryLocation loc)
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
}
