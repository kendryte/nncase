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
using Nncase.Passes.Analysis;
using Nncase.Passes.Mutators;
using Nncase.Passes.Transforms;
using Nncase.Targets;
using Nncase.TIR;

namespace Nncase.Passes;

public sealed class CPUTIRSelectionPass : TIRSelectionPass
{
    private readonly CompileOptions _compileOptions;

    public CPUTIRSelectionPass(CompileOptions compileOptions)
        : base(CPUTarget.Kind)
    {
        _compileOptions = compileOptions;
    }

    protected override Expr SelectCall(Call call, IReadOnlyList<BaseExpr> arguments, Expr output)
    {
        var op = call.Target;
        switch (op)
        {
            case IR.Math.Unary unary:
                return GenerateUnary(unary.UnaryOp, arguments, output);
            case IR.CustomCPU.Unary unary:
                return GenerateUnary(unary.UnaryOp, arguments, output);
            case IR.Math.Clamp clamp:
                return GenerateClamp(call, arguments, output);
            case IR.Distributed.Boxing boxing:
                return GenerateBoxing(call, boxing, arguments, output);
            case IR.Distributed.ForceBoxing forceBoxing:
                return T.Memcopy(output, (Expr)arguments[0]);
            case IR.Math.Binary binary:
                return GenerateBinary(binary.BinaryOp, arguments, output);
            case IR.CPU.Pack pack:
                return TIR.F.CPU.Pack((Expr)arguments[0], output, pack.Lanes, pack.Axes);
            case IR.CPU.Unpack unpack:
                return TIR.F.CPU.Unpack((Expr)arguments[0], output, unpack.Lanes, unpack.Axes);
            case IR.CPU.PackedBinary packedBinary:
                return TIR.F.CPU.Binary(packedBinary.BinaryOp, (Expr)arguments[0], (Expr)arguments[1], output);
            case IR.CPU.PackedMatMul packed_mat_mul_summa when GetArgumentType(arguments[0]) is DistributedType dta && dta.AxisPolices[^1] is SBPSplit:
                return TIR.F.CPU.SUMMA((Expr)arguments[0], (Expr)arguments[1], output, None.Default, packed_mat_mul_summa.LhsPackedAxes, packed_mat_mul_summa.LhsPadedNums, packed_mat_mul_summa.RhsPackedAxes, packed_mat_mul_summa.RhsPadedNums, packed_mat_mul_summa.TransposeA, packed_mat_mul_summa.TransposeB);
            case IR.Math.MatMul when GetArgumentType(arguments[0]) is DistributedType dta && dta.AxisPolices[^1] is SBPSplit:
                return TIR.F.CPU.SUMMA((Expr)arguments[0], (Expr)arguments[1], output, None.Default);
            case IR.CPU.PackedMatMul packedMatMul:
                return TIR.F.CPU.Matmul((Expr)arguments[0], (Expr)arguments[1], output, None.Default, packedMatMul.LhsPackedAxes, packedMatMul.LhsPadedNums, packedMatMul.RhsPackedAxes, packedMatMul.RhsPadedNums, packedMatMul.TransposeA, packedMatMul.TransposeB, packedMatMul.FusedReduce);
            case IR.Math.MatMul matmul:
                return TIR.F.CPU.Matmul((Expr)arguments[0], (Expr)arguments[1], output, None.Default);
            case IR.CustomCPU.MatMul matmul:
                return TIR.F.CPU.Matmul((Expr)arguments[0], (Expr)arguments[1], output, None.Default);
            case IR.NN.Conv2D conv:
                {
                    var input = call[IR.NN.Conv2D.Input];
                    var weights = call[IR.NN.Conv2D.Weights];
                    var bias = call[IR.NN.Conv2D.Bias];
                    var strides = ((TensorConst)call[IR.NN.Conv2D.Stride]).Value.ToArray<int>();
                    var padding = ((TensorConst)call[IR.NN.Conv2D.Padding]).Value.ToArray<int>();
                    var dilation = ((TensorConst)call[IR.NN.Conv2D.Dilation]).Value.ToArray<int>();
                    var groups = ((TensorConst)call[IR.NN.Conv2D.Groups]).Value.ToScalar<int>();
                    var fusedClamp = ((TensorConst)call[IR.NN.Conv2D.FusedClamp]).Value.ToArray<float>();
                    var wShape = weights.CheckedShape.ToValueArray();
                    var outShape = call.CheckedShape.ToValueArray();
                    if (fusedClamp[0] != float.NegativeInfinity || fusedClamp[1] != float.PositiveInfinity || conv.PadMode != PadMode.Constant)
                    {
                        throw new NotSupportedException("not support this conv2d");
                    }

                    return TIR.F.CPU.Conv2D((Expr)arguments[0], (Expr)arguments[1], (Expr)arguments[2], output, strides, padding, dilation, groups, conv.PadMode, call.CheckedType is DistributedType dt_conv ? dt_conv : null!);
                }

            case IR.CPU.Im2col im2col:
                return TIR.F.CPU.Im2col((Expr)arguments[0], output, im2col.Kernel, im2col.Stride, im2col.Padding, im2col.PackedAxes, im2col.PadedNums);
            case IR.Imaging.ResizeImage resize:
                if ((call[IR.Imaging.ResizeImage.Roi] is not None && ((RankedShape)call[IR.Imaging.ResizeImage.Roi].CheckedShape).Size != 0) || resize.IsTFResize)
                {
                    throw new NotSupportedException("not support tf resize");
                }

                return TIR.F.CPU.ResizeImage((Expr)arguments[0], output, Array.Empty<int>(), Array.Empty<int>(), ((TensorConst)call[IR.Imaging.ResizeImage.NewSize]).Value.ToArray<int>(), resize.ResizeMode, resize.TransformationMode, resize.NearestMode);
            case IR.CPU.ResizeImage resize:
                return TIR.F.CPU.ResizeImage((Expr)arguments[0], output, resize.PackedAxes.ToArray(), resize.PadedNums.ToArray(), resize.NewSize.ToArray(), resize.ResizeMode, resize.TransformationMode, resize.NearestMode);
            case IR.Tensors.Slice slice:
                return TIR.F.CPU.Slice((Expr)arguments[0], (Expr)arguments[1], (Expr)arguments[2], output, ((TensorConst)call[IR.Tensors.Slice.Axes]).Value.ToArray<int>(), ((TensorConst)call[IR.Tensors.Slice.Strides]).Value.ToArray<int>());
            case IR.Tensors.Concat concat:
                return TIR.F.CPU.Concat(((IR.Tuple)arguments[0]).Fields.AsValueEnumerable().Select(x => (Expr)x).ToArray(), output, concat.Axis);
            case IR.Tensors.Transpose trans:
                return TIR.F.CPU.Transpose((Expr)arguments[0], output, ((TensorConst)call[IR.Tensors.Transpose.Perm]).Value.ToArray<int>());
            case IR.NN.Swish swish:
                return TIR.F.CPU.Swish((Expr)arguments[0], output, ((TensorConst)call[IR.NN.Swish.Beta]).Value.ToScalar<float>());
            case IR.Tensors.Gather gather:
                return TIR.F.CPU.Gather((Expr)arguments[0], (Expr)arguments[1], output, gather.Axis);
            case IR.NN.Pad pad:
                return TIR.F.CPU.Pad((Expr)arguments[0], output, ((TensorConst)call[IR.NN.Pad.Pads]).Value.ToArray<int>(), ((TensorConst)call[IR.NN.Pad.Value]).Value.ToArray<float>()[0]);
            case IR.Math.Reduce reduce:
                return TIR.F.CPU.Reduce((Expr)arguments[0], output, false, Array.Empty<int>(), Array.Empty<int>(), ((TensorConst)call[IR.Math.Reduce.Axes]).Value.ToArray<int>().OrderBy(a => a).ToArray(), ((TensorConst)call[IR.Math.Reduce.KeepDims]).Value.ToArray<bool>()[0], reduce.ReduceOp);
            case IR.Math.ReduceArg reduceArg:
                return TIR.F.CPU.ReduceArg((Expr)arguments[0], output, ((TensorConst)call[IR.Math.ReduceArg.Axis]).Value.ToArray<int>()[0], ((TensorConst)call[IR.Math.ReduceArg.KeepDims]).Value.ToArray<bool>()[0], ((TensorConst)call[IR.Math.ReduceArg.SelectLastIndex]).Value.ToArray<bool>()[0], reduceArg.ReduceArgOp, reduceArg.DestType);
            case IR.Tensors.Cast cast:
                return TIR.F.CPU.Cast((Expr)arguments[0], output, cast.NewType, cast.CastMode);
            case IR.Tensors.Where where:
                return TIR.F.CPU.Where((Expr)arguments[0], (Expr)arguments[1], (Expr)arguments[2], output);
            case IR.Tensors.Expand expand:
                return TIR.F.CPU.Expand((Expr)arguments[0], output);
            case IR.NN.Erf erf:
                return TIR.F.CPU.Erf((Expr)arguments[0], output);
            case IR.CPU.PackedReduce pr:
                return TIR.F.CPU.Reduce((Expr)arguments[0], output, false, pr.PackedAxes.ToArray(), pr.PadedNums.ToArray(), pr.Axes, pr.KeepDims, pr.ReduceOp);
            case IR.Math.Compare compare:
                return TIR.F.CPU.Compare(compare.CompareOp, (Expr)arguments[0], (Expr)arguments[1], output);
            case IR.Tensors.GetItem getItem:
                return TIR.F.CPU.GetItem((Expr)arguments[0], (Expr)arguments[1], output);
            case IR.Tensors.Reshape:
                return TIR.F.CPU.Reshape((Expr)arguments[0], output);
            case IR.Tensors.ScatterND scatterND:
                return TIR.F.CPU.ScatterND((Expr)arguments[0], (Expr)arguments[1], (Expr)arguments[2], output);
            case IR.Tensors.Stack stack:
                return TIR.F.CPU.Stack(((IR.Tuple)arguments[0]).Fields.AsValueEnumerable().Select(x => (Expr)x).ToArray(), output, ((TensorConst)call[IR.Tensors.Stack.Axis]).Value.ToScalar<int>());
            case IR.Tensors.Unsqueeze:
                return TIR.F.CPU.Reshape((Expr)arguments[0], output);
            default:
                throw new NotSupportedException($"Not supported: {op}");
        }
    }

    private Expr GenerateUnary(UnaryOp unaryOp, IReadOnlyList<BaseExpr> arguments, Expr output)
    {
        var input = (Expr)arguments[IR.Math.Unary.Input.Index];
        return TIR.F.CPU.Unary(unaryOp, input, output);
    }

    private Expr GenerateBinary(BinaryOp binaryOp, IReadOnlyList<BaseExpr> arguments, Expr output)
    {
        return TIR.F.CPU.Binary(binaryOp, (Expr)arguments[0], (Expr)arguments[1], output);
    }

    private Expr GenerateClamp(Call call, IReadOnlyList<BaseExpr> arguments, Expr output)
    {
        var min = ((TensorConst)call[IR.Math.Clamp.Min]).Value.ToScalar<float>();
        var max = ((TensorConst)call[IR.Math.Clamp.Max]).Value.ToScalar<float>();
        return TIR.F.CPU.Clamp((Expr)arguments[0], output, min, max);
    }

    private Expr GenerateBoxing(Call call, IR.Distributed.Boxing boxing, IReadOnlyList<BaseExpr> arguments, Expr output)
    {
        switch (call[IR.Distributed.Boxing.Input].CheckedType, boxing.NewType)
        {
            case (TensorType, DistributedType distTensorType):
                return TIR.F.CPU.TensorLoad(output, (Expr)arguments[0], distTensorType.AxisPolices, distTensorType.Placement);
            case (DistributedType distTensorType, TensorType):
                return TIR.F.CPU.TensorStore((Expr)arguments[0], output, distTensorType.AxisPolices, distTensorType.Placement);
            case (DistributedType inType, DistributedType outType):
                return TIR.F.CPU.GatherReduceScatter((Expr)arguments[0], output, inType, outType);
            default:
                throw new NotSupportedException();
        }
    }
}
