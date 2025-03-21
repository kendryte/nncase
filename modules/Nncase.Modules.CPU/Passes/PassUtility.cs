// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NetFabric.Hyperlinq;
using Nncase.IR;

namespace Nncase.Passes;

public static class PassUtility
{
    public static bool IsCpuSupported(Op op)
    {
        if (op.GetType().Namespace == "Nncase.IR.CPU" || op.GetType().Namespace == "Nncase.IR.CustomCPU")
        {
            return true;
        }

        return op is IR.Distributed.Boxing
            or IR.Math.Unary
            or IR.Math.Binary { BinaryOp: BinaryOp.Add or BinaryOp.Sub or BinaryOp.Mul or BinaryOp.Div }
            or IR.Math.Clamp
            or IR.Math.Compare
            or IR.Math.MatMul
            or IR.Math.Reduce
            or IR.Math.ReduceArg
            or IR.Imaging.ResizeImage { IsTFResize: false }
            or IR.NN.Conv2D { PadMode: PadMode.Constant }
            or IR.NN.Erf
            or IR.NN.InstanceNormalization
            or IR.NN.LayerNorm
            or IR.NN.Pad { PadMode: PadMode.Constant }
            or IR.NN.Softmax
            or IR.NN.Swish
            or IR.Tensors.Cast
            or IR.Tensors.Concat
            or IR.Tensors.Expand
            or IR.Tensors.Gather
            or IR.Tensors.GetItem
            or IR.Tensors.Reshape
            or IR.Tensors.ScatterND
            or IR.Tensors.Slice
            or IR.Tensors.Stack
            or IR.Tensors.Transpose
            or IR.Tensors.Unsqueeze
            or IR.Tensors.Where;
    }

    public static bool IsCpuSupported(Op op, Call call, ReadOnlySpan<Expr> arguments, string moduleKind = "cpu")
    {
        if (!IsCpuSupported(op))
        {
            return false;
        }

        for (int i = 0; i < arguments.Length; i++)
        {
            var param = op.Parameters[i];
            var arg = arguments[i];
            if (arg.CheckedType switch { TensorType t => !t.Shape.IsRanked, _ => false })
            {
                return false;
            }
        }

        switch (op)
        {
            case IR.Imaging.ResizeImage:
                var roi = call[IR.Imaging.ResizeImage.Roi];
                if (roi is not IR.None && roi.CheckedShape.Rank != 0)
                {
                    return false;
                }

                break;
            case IR.Tensors.Slice slice:
                if (((TensorConst)call[IR.Tensors.Slice.Strides]).Value.Cast<long>().Any(s => s < 0))
                {
                    return false;
                }

                break;
            case IR.NN.Conv2D conv2d:
                if (((TensorConst)call[IR.NN.Conv2D.FusedClamp]).Value.Cast<float>() is var clamp)
                {
                    return clamp.SequenceEqual(new[] { float.NegativeInfinity, float.PositiveInfinity });
                }

                break;
            case IR.Math.Binary binary:
                if (arguments.AsValueEnumerable().Any(x => x.CheckedType is AnyType || x is If))
                {
                    return false;
                }

                break;
            case IR.NN.Pad pad:
                if (call[IR.NN.Pad.Pads] is not TensorConst)
                {
                    return false;
                }

                break;
            case IR.Math.Reduce reduce:
                var axis = ((TensorConst)arguments.ToArray()[1]).Value.ToArray<int>().OrderBy(x => x).ToArray();
                bool consecutiveAixs = axis.Length <= 1 || axis.Zip(axis.Skip(1)).All(p => p.First == p.Second - 1);
                if (reduce.ReduceOp == ReduceOp.Prod ||
                 arguments.ToArray()[0].CheckedDataType == DataTypes.Float16 ||
                 !consecutiveAixs)
                {
                    return false;
                }

                break;

            case IR.Tensors.Where where:
                if (arguments.ToArray()[0].CheckedShape != call.CheckedShape)
                {
                    return false;
                }

                break;
            case IR.Tensors.Gather gather:
                if (moduleKind == "xpu")
                {
                    return false;
                }

                break;
            default:
                break;
        }

        return true;
    }
}
