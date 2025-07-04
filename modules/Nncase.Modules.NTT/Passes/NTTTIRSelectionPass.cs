// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reactive;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using NetFabric.Hyperlinq;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.Shapes;
using Nncase.Passes.Analysis;
using Nncase.Passes.Mutators;
using Nncase.Passes.Transforms;
using Nncase.Targets;
using Nncase.TIR;

namespace Nncase.Passes;

public sealed class NTTTIRSelectionPass : TIRSelectionPass
{
    private readonly CompileOptions _compileOptions;

    public NTTTIRSelectionPass(CompileOptions compileOptions, string moduleKind = CPUTarget.Kind)
        : base(moduleKind)
    {
        _compileOptions = compileOptions;
    }

    protected override Expr SelectCall(Call call, IReadOnlyList<BaseExpr> arguments, ref Expr output)
    {
        var op = call.Target;
        switch (op)
        {
            case IR.Math.Unary unary:
                return GenerateUnary(unary.UnaryOp, arguments, output);
            case IR.CustomNTT.Unary unary:
                return GenerateUnary(unary.UnaryOp, arguments, output);
            case IR.Math.Clamp clamp:
                return GenerateClamp(call, arguments, output);
            case IR.Distributed.Boxing boxing:
                return GenerateBoxing(call, boxing, arguments, output);
            case IR.Distributed.ForceBoxing forceBoxing:
                return T.Memcopy(output, (Expr)arguments[0]);
            case IR.Math.Binary binary:
                return GenerateBinary(binary.BinaryOp, arguments, output);
            case IR.Tensors.Pack pack:
                return TIR.F.NTT.Pack((Expr)arguments[0], output, pack.Lanes, pack.Axes);
            case IR.Tensors.Unpack unpack:
                return TIR.F.NTT.Unpack((Expr)arguments[0], output, unpack.Lanes, unpack.Axes);
            case IR.NTT.PackedBinary packedBinary:
                return TIR.F.NTT.Binary(packedBinary.BinaryOp, (Expr)arguments[0], (Expr)arguments[1], output);
            case IR.NTT.PackedMatMul packed_mat_mul_summa when GetArgumentType(arguments[0]) is DistributedType dta && dta.AxisPolicies[^1] is SBPSplit:
                return TIR.F.NTT.SUMMA((Expr)arguments[0], (Expr)arguments[1], output, None.Default, packed_mat_mul_summa.LhsPackedAxes, packed_mat_mul_summa.RhsPackedAxes, packed_mat_mul_summa.TransposeA, packed_mat_mul_summa.TransposeB);
            case IR.Math.MatMul when GetArgumentType(arguments[0]) is DistributedType dta && dta.AxisPolicies[^1] is SBPSplit:
                return TIR.F.NTT.SUMMA((Expr)arguments[0], (Expr)arguments[1], output, None.Default);
            case IR.NTT.PackedMatMul packedMatMul:
                return TIR.F.NTT.Matmul((Expr)arguments[0], (Expr)arguments[1], output, None.Default, packedMatMul.LhsPackedAxes, packedMatMul.RhsPackedAxes, packedMatMul.TransposeA, packedMatMul.TransposeB, packedMatMul.FusedReduce);
            case IR.Math.MatMul matmul:
                return TIR.F.NTT.Matmul((Expr)arguments[0], (Expr)arguments[1], output, None.Default);
            case IR.CustomNTT.MatMul matmul:
                return TIR.F.NTT.Matmul((Expr)arguments[0], (Expr)arguments[1], output, None.Default, matmul.LhsPackedAxes, matmul.RhsPackedAxes, matmul.TransposeA, matmul.TransposeB, false, matmul.CSourcePath);
            case IR.NN.Conv2D conv:
                {
                    var input = call[IR.NN.Conv2D.Input];
                    var weights = call[IR.NN.Conv2D.Weights];
                    var bias = call[IR.NN.Conv2D.Bias];
                    var strides = ((RankedShape)call[IR.NN.Conv2D.Stride]).ToValueArray();
                    var padding = Tensor.From(((Paddings)call[IR.NN.Conv2D.Padding]).ToValueArray()).ToArray();
                    var dilation = ((RankedShape)call[IR.NN.Conv2D.Dilation]).ToValueArray();
                    var groups = ((Dimension)call[IR.NN.Conv2D.Groups]).FixedValue;
                    var fusedClamp = ((TensorConst)call[IR.NN.Conv2D.FusedClamp]).Value.ToArray<float>();
                    var wShape = weights.CheckedShape.ToValueArray();
                    var outShape = call.CheckedShape.ToValueArray();
                    if (fusedClamp[0] != float.NegativeInfinity || fusedClamp[1] != float.PositiveInfinity || conv.PadMode != PadMode.Constant)
                    {
                        throw new NotSupportedException("not support this conv2d");
                    }

                    return TIR.F.NTT.Conv2D((Expr)arguments[0], (Expr)arguments[1], (Expr)arguments[2], output, strides, padding, dilation, groups, conv.PadMode, call.CheckedType is DistributedType dt_conv ? dt_conv : null!);
                }

            case IR.NTT.Im2col im2col:
                return TIR.F.NTT.Im2col((Expr)arguments[0], output, im2col.Kernel, im2col.Stride, im2col.Padding, im2col.PackedAxes, im2col.PadedNums);
            case IR.Imaging.ResizeImage resize:
                if ((call[IR.Imaging.ResizeImage.Roi] is not None && ((RankedShape)call[IR.Imaging.ResizeImage.Roi].CheckedShape).Size != 0) || resize.IsTFResize)
                {
                    throw new NotSupportedException("not support tf resize");
                }

                return TIR.F.NTT.ResizeImage((Expr)arguments[0], output, Array.Empty<int>(), Array.Empty<Dimension>(), ((RankedShape)call[IR.Imaging.ResizeImage.NewSize]).ToValueArray().ToInts(), resize.ResizeMode, resize.TransformationMode, resize.NearestMode);
            case IR.NTT.ResizeImage resize:
                return TIR.F.NTT.ResizeImage((Expr)arguments[0], output, resize.PackedAxes.ToArray(), ((RankedShape)call[IR.NTT.ResizeImage.PadedNums]).Dimensions.ToArray(), resize.NewSize.ToArray(), resize.ResizeMode, resize.TransformationMode, resize.NearestMode);
            case IR.Tensors.Slice slice:
                return TIR.F.NTT.Slice((Expr)arguments[0], (RankedShape)arguments[1], (RankedShape)arguments[2], output, ((RankedShape)call[IR.Tensors.Slice.Axes]).ToValueArray().ToInts(), ((RankedShape)call[IR.Tensors.Slice.Strides]).ToValueArray().ToInts());
            case IR.Tensors.Concat concat:
                return TIR.F.NTT.Concat(((IR.Tuple)arguments[0]).Fields.AsValueEnumerable().Select(x => (Expr)x).ToArray(), output, concat.Axis);
            case IR.Tensors.Transpose trans:
                return TIR.F.NTT.Transpose((Expr)arguments[0], output, ((RankedShape)call[IR.Tensors.Transpose.Perm]).ToValueArray().ToInts());
            case IR.NN.Swish swish:
                return TIR.F.NTT.Swish((Expr)arguments[0], output, ((TensorConst)call[IR.NN.Swish.Beta]).Value.ToScalar<float>());
            case IR.Tensors.Gather gather:
                return TIR.F.NTT.Gather((Expr)arguments[0], (Expr)arguments[1], output, gather.Axis);
            case IR.NN.Pad pad when pad.PadMode == PadMode.Constant:
                return TIR.F.NTT.Pad((Expr)arguments[0], output, ((Paddings)call[IR.NN.Pad.Pads]).Select(p => new[] { p.Before, p.After }).SelectMany(p => p).ToArray(), ((TensorConst)call[IR.NN.Pad.Value]).Value.ToArray<float>()[0]);
            case IR.Math.Reduce reduce:
                return TIR.F.NTT.Reduce((Expr)arguments[0], output, false, Array.Empty<int>(), Array.Empty<Dimension>(), ((RankedShape)call[IR.Math.Reduce.Axes]).ToValueArray().Select(x => Util.PositiveIndex(x, arguments[0].CheckedTensorType)).OrderBy(a => a).ToArray().ToInts(), ((TensorConst)call[IR.Math.Reduce.KeepDims]).Value.ToArray<bool>()[0], reduce.ReduceOp);
            case IR.Math.ReduceArg reduceArg:
                return TIR.F.NTT.ReduceArg((Expr)arguments[0], output, (int)((DimConst)call[IR.Math.ReduceArg.Axis]).FixedValue, ((TensorConst)call[IR.Math.ReduceArg.KeepDims]).Value.ToArray<bool>()[0], ((TensorConst)call[IR.Math.ReduceArg.SelectLastIndex]).Value.ToArray<bool>()[0], reduceArg.ReduceArgOp, reduceArg.DestType);
            case IR.Tensors.Cast cast:
                return TIR.F.NTT.Cast((Expr)arguments[0], output, cast.NewType, cast.CastMode, cast.PackAxes);
            case IR.Tensors.Where where:
                return TIR.F.NTT.Where((Expr)arguments[0], (Expr)arguments[1], (Expr)arguments[2], output);
            case IR.Tensors.Expand expand:
                return TIR.F.NTT.Expand((Expr)arguments[0], output);
            case IR.NN.Erf erf:
                return TIR.F.NTT.Erf((Expr)arguments[0], output);
            case IR.NTT.PackedReduce pr:
                return TIR.F.NTT.Reduce((Expr)arguments[0], output, false, pr.PackedAxes.ToArray(), ((RankedShape)call[IR.NTT.PackedReduce.PadedNums]).Dimensions.ToArray(), pr.Axes, pr.KeepDims, pr.ReduceOp);
            case IR.Math.Compare compare:
                return TIR.F.NTT.Compare(compare.CompareOp, (Expr)arguments[0], (Expr)arguments[1], output);
            case IR.Tensors.GetItem getItem:
                return TIR.F.NTT.GetItem((Expr)arguments[0], arguments[1], output);
            case IR.Tensors.Reshape:
                return TIR.F.NTT.Reshape((Expr)arguments[0], output);
            case IR.Tensors.ScatterND scatterND:
                return TIR.F.NTT.ScatterND((Expr)arguments[0], (Expr)arguments[1], (Expr)arguments[2], output);
            case IR.Tensors.Stack stack:
                return TIR.F.NTT.Stack(((IR.Tuple)arguments[0]).Fields.AsValueEnumerable().Select(x => (Expr)x).ToArray(), output, ((TensorConst)call[IR.Tensors.Stack.Axis]).Value.ToScalar<int>());
            case IR.Tensors.Unsqueeze:
                return TIR.F.NTT.Reshape((Expr)arguments[0], output);
            case IR.NN.UpdatePagedAttentionKVCache upkv:
                output = (Expr)arguments[1];
                return TIR.F.NTT.UpdatePagedAttentionKVCache((Expr)arguments[0], (Expr)arguments[1], upkv.CacheKind, upkv.LayerId, upkv.Layout);
            case IR.NN.GatherPagedAttentionKVCache gakv:
                return TIR.F.NTT.GatherPagedAttentionKVCache((Expr)arguments[0], (Expr)arguments[1], output);
            case IR.NN.CreatePagedAttentionKVCache ctkv:
                return TIR.F.NTT.CreatePagedAttentionKVCache(ctkv.Config, (Expr)arguments[0], (Expr)arguments[1], (Expr)arguments[2], (Expr)arguments[3], (Expr)arguments[4], (Expr)arguments[5], (Expr)arguments[6], (Expr)arguments[7], output);
            case IR.NN.IdentityPagedAttentionKVCache ctkv:
                output = (Expr)arguments[0];
                return TIR.F.NTT.IdentityPagedAttentionKVCache((Expr)arguments[0], (Expr)arguments[1], (Expr)arguments[2], (Expr)arguments[3], (Expr)arguments[4], (Expr)arguments[5], (Expr)arguments[6], (Expr)arguments[7], (Expr)arguments[8]);
            case IR.NN.PagedAttention pgat:
                return TIR.F.NTT.PagedAttention((Expr)arguments[0], (Expr)arguments[1], (Expr)arguments[2], (Expr)arguments[3], pgat.LayerId, output, pgat.Layout, pgat.HiddenSize);
            case IR.Tensors.ConstantOfShape constantOfShape:
                return TIR.F.NTT.ConstantOfShape((Shape)arguments[0], (Expr)arguments[1], output);
            case IR.Tensors.Range range:
                return TIR.F.NTT.Range((Expr)arguments[0], (Expr)arguments[1], (Expr)arguments[2], output);
            case IR.Shapes.AsTensor asTensor:
                output = call;
                return call;
            case IR.NN.GetPositionIds getPositionIds:
                return TIR.F.NTT.GetPositionIds((Expr)arguments[1], output);
            case IR.NN.LayerNorm ln:
                return TIR.F.NTT.PackedLayerNorm((Expr)arguments[0], (Expr)arguments[1], (Expr)arguments[2], output, ln.Axis, ln.Epsilon, ln.UseMean, Array.Empty<int>(), Array.Empty<Dimension>());
            case IR.NTT.PackedLayerNorm ln:
                return TIR.F.NTT.PackedLayerNorm((Expr)arguments[0], (Expr)arguments[1], (Expr)arguments[2], output, ln.Axis, ln.Epsilon, ln.UseMean, ln.PackedAxes, ((RankedShape)call[IR.NTT.PackedLayerNorm.PadedNums]).Dimensions.ToArray());
            case IR.NN.Softmax softmax:
                return TIR.F.NTT.PackedSoftmax((Expr)arguments[0], output, (int)((DimConst)call[IR.NN.Softmax.Axis]).FixedValue, Array.Empty<int>());
            case IR.NTT.PackedSoftmax softmax:
                return TIR.F.NTT.PackedSoftmax((Expr)arguments[0], output, softmax.Axis, softmax.PackedAxes);
            default:
                throw new NotSupportedException($"Not supported: {op}");
        }
    }

    private Expr GenerateUnary(UnaryOp unaryOp, IReadOnlyList<BaseExpr> arguments, Expr output)
    {
        var input = (Expr)arguments[IR.Math.Unary.Input.Index];
        return TIR.F.NTT.Unary(unaryOp, input, output);
    }

    private Expr GenerateBinary(BinaryOp binaryOp, IReadOnlyList<BaseExpr> arguments, Expr output)
    {
        return TIR.F.NTT.Binary(binaryOp, (Expr)arguments[0], (Expr)arguments[1], output);
    }

    private Expr GenerateClamp(Call call, IReadOnlyList<BaseExpr> arguments, Expr output)
    {
        var min = ((TensorConst)call[IR.Math.Clamp.Min]).Value.ToScalar<float>();
        var max = ((TensorConst)call[IR.Math.Clamp.Max]).Value.ToScalar<float>();
        return TIR.F.NTT.Clamp((Expr)arguments[0], output, min, max);
    }

    private Expr GenerateBoxing(Call call, IR.Distributed.Boxing boxing, IReadOnlyList<BaseExpr> arguments, Expr output)
    {
        switch (call[IR.Distributed.Boxing.Input].CheckedType, boxing.NewType)
        {
            case (TensorType, DistributedType distTensorType):
                return TIR.F.NTT.TensorLoad(output, (Expr)arguments[0], distTensorType.AxisPolicies, distTensorType.Placement);
            case (DistributedType distTensorType, TensorType):
                return TIR.F.NTT.TensorStore((Expr)arguments[0], output, distTensorType.AxisPolicies, distTensorType.Placement);
            case (DistributedType inType, DistributedType outType):
                return TIR.F.NTT.GatherReduceScatter((Expr)arguments[0], output, inType, outType);
            default:
                throw new NotSupportedException();
        }
    }
}
