// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.Tensors;

namespace Nncase.Importer
{
    public class Glm4V9B : HuggingFaceModel
    {
        public override Call RotateHalf(Expr x)
        {
            var x1 = IR.F.Tensors.Slice(
                x,
                new[] { 0L },
                new[] { -1L },
                new[] { -1L },
                new[] { 2L });
            var x2 = IR.F.Tensors.Slice(
                x,
                new[] { 1L },
                new[] { -1L },
                new[] { -1L },
                new[] { 2L });

            return IR.F.Tensors.Flatten(IR.F.Tensors.Stack(new IR.Tuple(IR.F.Math.Neg(x2), x1), -1), -2);
        }

        public virtual System.Tuple<Call, Call> ApplyRotaryPosEmb(Expr q, Expr k, Expr cos, Expr sin, int unSqueezeDim = 1)
        {
            cos = IR.F.Tensors.Unsqueeze(cos, new[] { unSqueezeDim });
            sin = IR.F.Tensors.Unsqueeze(sin, new[] { unSqueezeDim });

            var cosShape = IR.F.Tensors.ShapeOf(cos);
            var sinShape = IR.F.Tensors.ShapeOf(sin);

            var cosFrontDims = IR.F.Tensors.Slice(cosShape, new[] { 0L }, new[] { -2L }, 0, 1);
            var sinFrontDims = IR.F.Tensors.Slice(sinShape, new[] { 0L }, new[] { -2L }, 0, 1);
            var cosLastDim = cosShape[-1];
            var sinLastDim = sinShape[-1];

            // repeat_interleave for cos & sin
            cos = IR.F.Tensors.Slice(cos, new[] { 0L }, cosLastDim / 2L, new[] { -1L }, new[] { 1L });
            var tmpCosShape = IR.F.Tensors.Concat(new IR.Tuple(cosFrontDims, -1), -1);
            var newCosShape = IR.F.Tensors.Concat(new IR.Tuple(cosFrontDims, -1), 2);
            cos = IR.F.Tensors.Reshape(cos, tmpCosShape);
            cos = IR.F.Tensors.Broadcast(cos, newCosShape);
            sin = IR.F.Tensors.Slice(sin, new[] { 0L }, sinLastDim / 2L, new[] { -1L }, new[] { 1L });
            var tmpSinShape = IR.F.Tensors.Concat(new IR.Tuple(sinFrontDims, -1), -1);
            var newSinShape = IR.F.Tensors.Concat(new IR.Tuple(sinFrontDims, -1), 2);
            cos = IR.F.Tensors.Reshape(cos, tmpSinShape);
            cos = IR.F.Tensors.Broadcast(cos, newSinShape);

            // 获取rotary_dim
            var rotaryDim = cosLastDim;

            // 将q和k分为两部分
            var qRot = IR.F.Tensors.Slice(q, new[] { 0L }, rotaryDim, new[] { -1L }, new[] { 1L });
            var qPass = IR.F.Tensors.Slice(q, rotaryDim, new[] { -1L }, new[] { -1L }, new[] { 1L });
            var kRot = IR.F.Tensors.Slice(k, new[] { 0L }, rotaryDim, new[] { -1L }, new[] { 1L });
            var kPass = IR.F.Tensors.Slice(k, rotaryDim, new[] { -1L }, new[] { -1L }, new[] { 1L });

            // 应用旋转位置嵌入
            var qRotEmbed = IR.F.Math.Binary(
                BinaryOp.Add,
                IR.F.Math.Binary(BinaryOp.Mul, qRot, cos),
                IR.F.Math.Binary(BinaryOp.Mul, RotateHalf(qRot), sin));
            var kRotEmbed = IR.F.Math.Binary(
                BinaryOp.Add,
                IR.F.Math.Binary(BinaryOp.Mul, kRot, cos),
                IR.F.Math.Binary(BinaryOp.Mul, RotateHalf(kRot), sin));

            // 拼接回完整形状
            var qEmbed = IR.F.Tensors.Concat(new IR.Tuple(qRotEmbed, qPass), -1);
            var kEmbed = IR.F.Tensors.Concat(new IR.Tuple(kRotEmbed, kPass), -1);

            return System.Tuple.Create(qEmbed, kEmbed);
        }
    }
}
