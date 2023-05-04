// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.IR;
using Nncase.IR.Tensors;
using Onnx;
using F = Nncase.IR.F;

namespace Nncase.Importer
{
    public partial class OnnxImporter
    {
        private Expr VisitExpand(in NodeProto op)
        {
            var (input, shape) = GetInputExprs(op, 0, 1);

            if (shape is TensorConst)
            {
                var maxLen = System.Math.Max(input.CheckedShape.Rank, shape.CheckedShape.Size);
                var outputShape = new Expr[maxLen];
                var inputShape = F.Tensors.ShapeOf(input);
                for (var i = 0; i < maxLen; i++)
                {
                    if (maxLen == input.CheckedShape.Rank)
                    {
                        outputShape[i] = F.Math.Max(inputShape[i], i < input.CheckedShape.Rank - shape.CheckedShape.Size ? 1L : shape[i - (input.CheckedShape.Rank - shape.CheckedShape.Size)]);
                    }
                    else
                    {
                        outputShape[i] = F.Math.Max(i < shape.CheckedShape.Size - input.CheckedShape.Rank ? 1L : inputShape[i - (shape.CheckedShape.Size - input.CheckedShape.Rank)], shape[i]);
                    }
                }

                return F.Tensors.Expand(input, F.Tensors.Stack(new IR.Tuple(outputShape), 0));
            }

            var rhs = input.CheckedDataType switch
            {
                var x when x == DataTypes.UInt8 => F.Tensors.ConstantOfShape(shape, (byte)1),
                var x when x == DataTypes.UInt16 => F.Tensors.ConstantOfShape(shape, (ushort)1),
                var x when x == DataTypes.UInt32 => F.Tensors.ConstantOfShape(shape, (uint)1),
                var x when x == DataTypes.UInt64 => F.Tensors.ConstantOfShape(shape, (ulong)1),
                var x when x == DataTypes.Int8 => F.Tensors.ConstantOfShape(shape, (sbyte)1),
                var x when x == DataTypes.Int16 => F.Tensors.ConstantOfShape(shape, (short)1),
                var x when x == DataTypes.Int32 => F.Tensors.ConstantOfShape(shape, (int)1),
                var x when x == DataTypes.Int64 => F.Tensors.ConstantOfShape(shape, (long)1),
                var x when x == DataTypes.Float16 => F.Tensors.ConstantOfShape(shape, (Half)1),
                var x when x == DataTypes.Float32 => F.Tensors.ConstantOfShape(shape, (float)1),
                var x when x == DataTypes.Float64 => F.Tensors.ConstantOfShape(shape, (double)1),
                var x when x == DataTypes.BFloat16 => F.Tensors.ConstantOfShape(shape, (BFloat16)1),
                var x when x == DataTypes.Boolean => F.Tensors.ConstantOfShape(shape, true),
                _ => throw new NotSupportedException("not supported expand type"),
            };
            return F.Tensors.Expand(input, F.Tensors.ShapeOf(F.Math.Mul(input, rhs)));
        }
    }
}
