// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.Tensors;

namespace Nncase.Importer
{
    public class Glm4V9B : HuggingFaceModel
    {
        private ModelInitContext? _context;

        public override void Initialize(ModelInitContext context, string dir)
        {
            base.Initialize(context, dir);
            _context = context;
        }

        public override Call RotateHalf(Expr x)
        {
            var xShape = IR.F.Tensors.ShapeOf(x);
            x = IR.F.Tensors.Reshape(x, new Shape(xShape[0].AsDim(), xShape[1].AsDim(), xShape[2].AsDim(), xShape[3].AsDim() / 2L, 2L));
            var x1 = IR.F.Tensors.Slice(
                x,
                new[] { 0L },
                new[] { 1L },
                new[] { -1L },
                new[] { 1L });
            var x2 = IR.F.Tensors.Slice(
                x,
                new[] { 1L },
                new[] { 2L },
                new[] { -1L },
                new[] { 1L });

            return IR.F.Tensors.Reshape(IR.F.Tensors.Concat(new IR.Tuple(IR.F.Math.Neg(x2), x1), -1), xShape);
        }

        public override System.Tuple<Call, Call> ApplyRotaryPosEmb(Expr q, Expr k, Expr cos, Expr sin, long unSqueezeDim = 1)
        {
            cos = IR.F.Tensors.Unsqueeze(cos, Tensor.From<long>(new long[] { 1 }));
            sin = IR.F.Tensors.Unsqueeze(sin, Tensor.From<long>(new long[] { 1 }));

            var cosShape = IR.F.Tensors.ShapeOf(cos);
            var sinShape = IR.F.Tensors.ShapeOf(sin);

            var cosLastDim = new Shape(cosShape[-1]);
            var sinLastDim = new Shape(sinShape[-1]);

            // repeat_interleave for cos & sin
            cos = IR.F.Tensors.Slice(cos, new[] { 0L }, cosLastDim / 2L, new[] { 3L }, new[] { 1L });
            cos = IR.F.Tensors.Unsqueeze(cos, -1);
            cos = IR.F.Tensors.Concat(new IR.Tuple(cos, cos), -1);
            cos = IR.F.Tensors.Reshape(cos, cosShape);

            sin = IR.F.Tensors.Slice(sin, new[] { 0L }, sinLastDim / 2L, new[] { 3L }, new[] { 1L });
            sin = IR.F.Tensors.Unsqueeze(sin, -1);
            sin = IR.F.Tensors.Concat(new IR.Tuple(sin, sin), -1);
            sin = IR.F.Tensors.Reshape(sin, sinShape);

            var rotaryDim = new Shape(IR.F.Tensors.ShapeOf(cos)[-1]);

            var qRot = IR.F.Tensors.Slice(q, new[] { 0L }, rotaryDim, new[] { 3L }, new[] { 1L });
            var qPass = IR.F.Tensors.Slice(q, rotaryDim, new Shape(IR.F.Tensors.ShapeOf(q)[-1]), new[] { 3L }, new[] { 1L });
            var kRot = IR.F.Tensors.Slice(k, new[] { 0L }, rotaryDim, new[] { 3L }, new[] { 1L });
            var kPass = IR.F.Tensors.Slice(k, rotaryDim, new Shape(IR.F.Tensors.ShapeOf(k)[-1]), new[] { 3L }, new[] { 1L });

            var qRotEmbed = IR.F.Math.Binary(
                BinaryOp.Add,
                IR.F.Math.Binary(BinaryOp.Mul, qRot, cos),
                IR.F.Math.Binary(BinaryOp.Mul, RotateHalf(qRot), sin));
            var kRotEmbed = IR.F.Math.Binary(
                BinaryOp.Add,
                IR.F.Math.Binary(BinaryOp.Mul, kRot, cos), // 1,1,16,128
                IR.F.Math.Binary(BinaryOp.Mul, RotateHalf(kRot), sin));

            var qEmbed = IR.F.Tensors.Concat(new IR.Tuple(qRotEmbed, qPass), -1);
            var kEmbed = IR.F.Tensors.Concat(new IR.Tuple(kRotEmbed, kPass), -1);
            return System.Tuple.Create(qEmbed, kEmbed);
        }

        public override Call LLMMlp(int count, Expr hiddenStates)
        {
            var gateUpProjW = _context!.ConstTensors![$"model.layers.{count}.mlp.gate_up_proj.weight"];
            var downProjW = _context.ConstTensors![$"model.layers.{count}.mlp.down_proj.weight"];

            var upStates = Linear(hiddenStates, gateUpProjW);
            var upStatesShape = new Shape(IR.F.Tensors.ShapeOf(upStates)[-1]);

            // gate, up_states = up_states.chunk(2, dim = -1)
            // dim == -1 mean can slice 1/2
            var gate = IR.F.Tensors.Slice(upStates, new[] { 0L }, upStatesShape / 2L, new[] { -1L }, new[] { 1L });
            upStates = IR.F.Tensors.Slice(upStates, upStatesShape / 2L, new Shape(IR.F.Tensors.ShapeOf(upStates)[-1]), new[] { -1L }, new[] { 1L });

            if (_context!.Config!.ContainsKey("hidden_act"))
            {
                var actType = _context!.Config!.GetNestedValue<string>("hidden_act");
                gate = ModelUtils.ActFunc(gate, actType);
            }

            upStates = upStates * gate;

            return Linear(upStates, downProjW);
        }
    }
}
