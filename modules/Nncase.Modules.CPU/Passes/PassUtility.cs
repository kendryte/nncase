﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.Passes;

public static class PassUtility
{
    public static bool IsCpuSupported(Op op)
    {
        if (op.GetType().Namespace == "Nncase.IR.CPU")
        {
            return true;
        }

        return op is IR.Math.Unary or IR.Math.MatMul or IR.NN.Conv2D { PadMode: PadMode.Constant } or IR.NN.Softmax or IR.NN.LayerNorm or IR.NN.InstanceNormalization or IR.Imaging.ResizeImage { IsTFResize: false } or IR.Tensors.Unsqueeze or IR.Tensors.Reshape or IR.Tensors.Slice or IR.Tensors.Concat or IR.Tensors.Transpose or IR.NN.Swish or IR.Tensors.Gather or IR.NN.Pad { PadMode: PadMode.Constant } or IR.Math.Reduce;
    }

    public static bool IsCpuSupported(Op op, IEnumerable<Expr> arguments)
    {
        if (!IsCpuSupported(op))
        {
            return false;
        }

        if (!op.Parameters.Zip(arguments).All(p => p.First.ParameterKind == ParameterKind.Input || (p.First.ParameterKind == ParameterKind.Attribute && p.Second is TensorConst)))
        {
            return false;
        }

        switch (op)
        {
            case IR.Imaging.ResizeImage:
                if (arguments.Skip(IR.Imaging.ResizeImage.Roi.Index).First() is not IR.None)
                {
                    return false;
                }

                break;
            case IR.Tensors.Slice slice:
                if (((TensorConst)arguments.Skip(IR.Tensors.Slice.Strides.Index).First()).Value.ToArray<int>().Any(s => s < 0))
                {
                    return false;
                }

                if (((TensorConst)arguments.Skip(IR.Tensors.Slice.Begins.Index).First()).Value.ToArray<int>().Any(s => s < 0))
                {
                    return false;
                }

                if (((TensorConst)arguments.Skip(IR.Tensors.Slice.Ends.Index).First()).Value.ToArray<int>().Any(s => s < 0))
                {
                    return false;
                }

                break;
            case IR.NN.Conv2D conv2d:
                if (((TensorConst)arguments.Skip(IR.NN.Conv2D.FusedClamp.Index).First()).Value.ToArray<float>() is var clamp)
                {
                    return clamp == new[] { float.NegativeInfinity, float.PositiveInfinity };
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
            default:
                break;
        }

        return true;
    }
}
