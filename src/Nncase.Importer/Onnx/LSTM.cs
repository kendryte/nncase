// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using Nncase.IR;
using Onnx;
using static Nncase.IR.F.RNN;
using static Nncase.IR.F.Tensors;
using static Nncase.LSTMHelper;
using Tuple = Nncase.IR.Tuple;

namespace Nncase.Importer
{
    public partial class OnnxImporter
    {
        private Expr VisitLSTM(in NodeProto op)
        {
            var actAlpha = GetOptionFloatsAttribute(op, "activation_alpha").Or(Array.Empty<float>());
            var actBeta = GetOptionFloatsAttribute(op, "activation_beta").Or(Array.Empty<float>());
            var clip = GetOptionFloatAttribute(op, "clip").Or(float.NaN);
            var direction = GetOptionStringAttribute(op, "direction").Or("forward");
            var numBirections = direction == "bidirectional" ? 2 : 1;
            var acts = GetOptionStringsAttribute(op, "activations").Or(new[] { "sigmoid", "tanh", "tanh" });
            if (numBirections == 2 && acts.Length == 3)
            {
                acts = acts.Concat(acts).ToArray();
            }

            // if 0
            // X.shape = [seq_length, batch_size, input_size]
            // Y(Outputs).shape = [seq_length, num_directions, batch_size, hidden_size]
            // initial_h.shape = Y_h.shape = initial_c.shape = Y_c.shape = [num_directions, batch_size, hidden_size]

            // If 1
            // X.shape = [batch_size, seq_length, input_size]
            // Y.shape = [batch_size, seq_length, num_directions, hidden_size]
            // initial_h.shape = Y_h.shape = initial_c.shape = Y_c.shape = [batch_size, num_directions, hidden_size].
            var layout = GetOptionIntAttribute(op, "layout").Or(0);
            var seqIndex = layout == 0 ? 0 : 1;
            var batchIndex = layout == 0 ? 1 : 0;

            // var T = GetInputDataType(op, 0);
            var (x, w) = GetInputExprs(op, 0, 1);
            x.InferenceType();
            var t = x.CheckedDataType;

            var hiddenSize = GetOptionIntAttribute(op, "hidden_size")
                    .Match(
                        x => (Expr)x,
                        () => Cast(Util.ShapeIndex(w, 1) / 4, DataTypes.Int64));
            var inputForget = GetOptionIntAttribute(op, "input_forget").Or(0);
            var seqLens = Cast(Util.ShapeIndex(x, seqIndex), DataTypes.Int64);
            var batchSize = Cast(Util.ShapeIndex(x, batchIndex), DataTypes.Int64);
            var r = GetInputExpr(op, 2);

            // cast to x type
            var b = GetOptionInputExpr(op, 3).Or(
                ExpandToType(0, t, numBirections, 8L * hiddenSize));

            // onnx type constraints sequence_lens: int32
            var sequenceLens = GetOptionInputExpr(op, 4).Or(
                ExpandToType(seqLens, DataTypes.Int32, batchSize));
            var initDefaultValue = ExpandToType(0, t, numBirections, batchSize, hiddenSize);
            var initialH = GetOptionInputExpr(op, 5).Or(initDefaultValue);
            var initialC = GetOptionInputExpr(op, 6).Or(initDefaultValue);
            var p = GetOptionInputExpr(op, 7).Or(
                ExpandToType(0, t, numBirections, 3L * hiddenSize));

            var outputCount = op.Output.Count;
            return LSTM(
                ToLSTMDirection(direction),
                ToLSTMLayout(layout),
                acts,
                x,
                w,
                r,
                b,
                sequenceLens,
                initialH,
                initialC,
                p,
                actAlpha,
                actBeta,
                clip,
                hiddenSize,
                inputForget,
                outputCount);
        }

        private Expr ExpandToType(Expr input, DataType t, params Expr[] dims)
        {
            return Cast(
                Expand(input, Stack(new Tuple(dims.Select(x => Cast(x, DataTypes.Int64)).ToArray()), 0)),
                t);
        }
    }
}
