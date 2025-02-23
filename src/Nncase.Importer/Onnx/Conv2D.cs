// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Nncase.IR;
using Onnx;
using static Nncase.IR.F.Tensors;
using F = Nncase.IR.F;

namespace Nncase.Importer
{
    public partial class OnnxGraphImporter
    {
        private Expr VisitConv2D(in NodeProto op)
        {
            var (input, weights) = GetInputExprs(op, 0, 1);
            var bias = GetBias(op, weights);
            var autoPad = GetStringAttribute(op, "auto_pad", "NOTSET");
            var dilation = GetDilationsAttribute(op).ToList();
            var group = GetIntAttribute(op, "group", 1);

            // if not present, should be inferred from input W
            var strides = GetStrideAttribute(op).ToArray<long>().ToList();

            var isConv1D = IsConv1D(weights);
            List<string> wOutputNames = new() { weights.Metadata.OutputNames![0] };
            if (isConv1D)
            {
                dilation.Insert(0, 1);
                strides.Insert(0, 1);
                input = To4D(input);
                weights = To4D(weights);
            }

            weights.Metadata.OutputNames = wOutputNames;
            var pads = AutoPad(op, autoPad, input, weights, strides.ToArray<long>(), dilation.ToArray(), isConv1D);
            pads.InferenceType();

            // if (weights.Metadata.OutputNames?[0] == "onnx::Conv_959" || weights.Metadata.OutputNames?[0] == "onnx::Conv_955" || weights.Metadata.OutputNames?[0] == "onnx::Conv_947")
            if (false)
            {
                // var biasOrg = ((TensorConst)bias).Value.ToArray<float>().Select(b => b / 64.0f).ToArray();
                // var newBias = Tensor.From(biasOrg, bias.CheckedShape);

                var biasOrg = ((TensorConst)bias).Value.ToArray<float>().Select(b => 0.0f).ToArray();
                var newBias = Tensor.From(biasOrg, bias.CheckedShape);

                var input0 = F.Tensors.Slice(input, new int[] { 0 }, new int[] { 1 }, new int[] { 1 }, new int[] { 1 });
                var input1 = F.Tensors.Slice(input, new int[] { 1 }, new int[] { 2 }, new int[] { 1 }, new int[] { 1 });
                var input2 = F.Tensors.Slice(input, new int[] { 2 }, new int[] { 3 }, new int[] { 1 }, new int[] { 1 });
                var input3 = F.Tensors.Slice(input, new int[] { 3 }, new int[] { 4 }, new int[] { 1 }, new int[] { 1 });
                var input4 = F.Tensors.Slice(input, new int[] { 4 }, new int[] { 5 }, new int[] { 1 }, new int[] { 1 });
                var input5 = F.Tensors.Slice(input, new int[] { 5 }, new int[] { 6 }, new int[] { 1 }, new int[] { 1 });
                var input6 = F.Tensors.Slice(input, new int[] { 6 }, new int[] { 7 }, new int[] { 1 }, new int[] { 1 });
                var input7 = F.Tensors.Slice(input, new int[] { 7 }, new int[] { 8 }, new int[] { 1 }, new int[] { 1 });
                var input8 = F.Tensors.Slice(input, new int[] { 8 }, new int[] { 9 }, new int[] { 1 }, new int[] { 1 });
                var input9 = F.Tensors.Slice(input, new int[] { 9 }, new int[] { 10 }, new int[] { 1 }, new int[] { 1 });
                var input10 = F.Tensors.Slice(input, new int[] { 10 }, new int[] { 11 }, new int[] { 1 }, new int[] { 1 });
                var input11 = F.Tensors.Slice(input, new int[] { 11 }, new int[] { 12 }, new int[] { 1 }, new int[] { 1 });
                var input12 = F.Tensors.Slice(input, new int[] { 12 }, new int[] { 13 }, new int[] { 1 }, new int[] { 1 });
                var input13 = F.Tensors.Slice(input, new int[] { 13 }, new int[] { 14 }, new int[] { 1 }, new int[] { 1 });
                var input14 = F.Tensors.Slice(input, new int[] { 14 }, new int[] { 15 }, new int[] { 1 }, new int[] { 1 });
                var input15 = F.Tensors.Slice(input, new int[] { 15 }, new int[] { 16 }, new int[] { 1 }, new int[] { 1 });
                var input16 = F.Tensors.Slice(input, new int[] { 16 }, new int[] { 17 }, new int[] { 1 }, new int[] { 1 });
                var input17 = F.Tensors.Slice(input, new int[] { 17 }, new int[] { 18 }, new int[] { 1 }, new int[] { 1 });
                var input18 = F.Tensors.Slice(input, new int[] { 18 }, new int[] { 19 }, new int[] { 1 }, new int[] { 1 });
                var input19 = F.Tensors.Slice(input, new int[] { 19 }, new int[] { 20 }, new int[] { 1 }, new int[] { 1 });
                var input20 = F.Tensors.Slice(input, new int[] { 20 }, new int[] { 21 }, new int[] { 1 }, new int[] { 1 });
                var input21 = F.Tensors.Slice(input, new int[] { 21 }, new int[] { 22 }, new int[] { 1 }, new int[] { 1 });
                var input22 = F.Tensors.Slice(input, new int[] { 22 }, new int[] { 23 }, new int[] { 1 }, new int[] { 1 });
                var input23 = F.Tensors.Slice(input, new int[] { 23 }, new int[] { 24 }, new int[] { 1 }, new int[] { 1 });
                var input24 = F.Tensors.Slice(input, new int[] { 24 }, new int[] { 25 }, new int[] { 1 }, new int[] { 1 });
                var input25 = F.Tensors.Slice(input, new int[] { 25 }, new int[] { 26 }, new int[] { 1 }, new int[] { 1 });
                var input26 = F.Tensors.Slice(input, new int[] { 26 }, new int[] { 27 }, new int[] { 1 }, new int[] { 1 });
                var input27 = F.Tensors.Slice(input, new int[] { 27 }, new int[] { 28 }, new int[] { 1 }, new int[] { 1 });
                var input28 = F.Tensors.Slice(input, new int[] { 28 }, new int[] { 29 }, new int[] { 1 }, new int[] { 1 });
                var input29 = F.Tensors.Slice(input, new int[] { 29 }, new int[] { 30 }, new int[] { 1 }, new int[] { 1 });
                var input30 = F.Tensors.Slice(input, new int[] { 30 }, new int[] { 31 }, new int[] { 1 }, new int[] { 1 });
                var input31 = F.Tensors.Slice(input, new int[] { 31 }, new int[] { 32 }, new int[] { 1 }, new int[] { 1 });
                var input32 = F.Tensors.Slice(input, new int[] { 32 }, new int[] { 33 }, new int[] { 1 }, new int[] { 1 });
                var input33 = F.Tensors.Slice(input, new int[] { 33 }, new int[] { 34 }, new int[] { 1 }, new int[] { 1 });
                var input34 = F.Tensors.Slice(input, new int[] { 34 }, new int[] { 35 }, new int[] { 1 }, new int[] { 1 });
                var input35 = F.Tensors.Slice(input, new int[] { 35 }, new int[] { 36 }, new int[] { 1 }, new int[] { 1 });
                var input36 = F.Tensors.Slice(input, new int[] { 36 }, new int[] { 37 }, new int[] { 1 }, new int[] { 1 });
                var input37 = F.Tensors.Slice(input, new int[] { 37 }, new int[] { 38 }, new int[] { 1 }, new int[] { 1 });
                var input38 = F.Tensors.Slice(input, new int[] { 38 }, new int[] { 39 }, new int[] { 1 }, new int[] { 1 });
                var input39 = F.Tensors.Slice(input, new int[] { 39 }, new int[] { 40 }, new int[] { 1 }, new int[] { 1 });
                var input40 = F.Tensors.Slice(input, new int[] { 40 }, new int[] { 41 }, new int[] { 1 }, new int[] { 1 });
                var input41 = F.Tensors.Slice(input, new int[] { 41 }, new int[] { 42 }, new int[] { 1 }, new int[] { 1 });
                var input42 = F.Tensors.Slice(input, new int[] { 42 }, new int[] { 43 }, new int[] { 1 }, new int[] { 1 });
                var input43 = F.Tensors.Slice(input, new int[] { 43 }, new int[] { 44 }, new int[] { 1 }, new int[] { 1 });
                var input44 = F.Tensors.Slice(input, new int[] { 44 }, new int[] { 45 }, new int[] { 1 }, new int[] { 1 });
                var input45 = F.Tensors.Slice(input, new int[] { 45 }, new int[] { 46 }, new int[] { 1 }, new int[] { 1 });
                var input46 = F.Tensors.Slice(input, new int[] { 46 }, new int[] { 47 }, new int[] { 1 }, new int[] { 1 });
                var input47 = F.Tensors.Slice(input, new int[] { 47 }, new int[] { 48 }, new int[] { 1 }, new int[] { 1 });
                var input48 = F.Tensors.Slice(input, new int[] { 48 }, new int[] { 49 }, new int[] { 1 }, new int[] { 1 });
                var input49 = F.Tensors.Slice(input, new int[] { 49 }, new int[] { 50 }, new int[] { 1 }, new int[] { 1 });
                var input50 = F.Tensors.Slice(input, new int[] { 50 }, new int[] { 51 }, new int[] { 1 }, new int[] { 1 });
                var input51 = F.Tensors.Slice(input, new int[] { 51 }, new int[] { 52 }, new int[] { 1 }, new int[] { 1 });
                var input52 = F.Tensors.Slice(input, new int[] { 52 }, new int[] { 53 }, new int[] { 1 }, new int[] { 1 });
                var input53 = F.Tensors.Slice(input, new int[] { 53 }, new int[] { 54 }, new int[] { 1 }, new int[] { 1 });
                var input54 = F.Tensors.Slice(input, new int[] { 54 }, new int[] { 55 }, new int[] { 1 }, new int[] { 1 });
                var input55 = F.Tensors.Slice(input, new int[] { 55 }, new int[] { 56 }, new int[] { 1 }, new int[] { 1 });
                var input56 = F.Tensors.Slice(input, new int[] { 56 }, new int[] { 57 }, new int[] { 1 }, new int[] { 1 });
                var input57 = F.Tensors.Slice(input, new int[] { 57 }, new int[] { 58 }, new int[] { 1 }, new int[] { 1 });
                var input58 = F.Tensors.Slice(input, new int[] { 58 }, new int[] { 59 }, new int[] { 1 }, new int[] { 1 });
                var input59 = F.Tensors.Slice(input, new int[] { 59 }, new int[] { 60 }, new int[] { 1 }, new int[] { 1 });
                var input60 = F.Tensors.Slice(input, new int[] { 60 }, new int[] { 61 }, new int[] { 1 }, new int[] { 1 });
                var input61 = F.Tensors.Slice(input, new int[] { 61 }, new int[] { 62 }, new int[] { 1 }, new int[] { 1 });
                var input62 = F.Tensors.Slice(input, new int[] { 62 }, new int[] { 63 }, new int[] { 1 }, new int[] { 1 });
                var input63 = F.Tensors.Slice(input, new int[] { 63 }, new int[] { 64 }, new int[] { 1 }, new int[] { 1 });


                var weights0 = F.Tensors.Slice(weights, new int[] { 0 }, new int[] { 1 }, new int[] { 1 }, new int[] { 1 });
                var weights1 = F.Tensors.Slice(weights, new int[] { 1 }, new int[] { 2 }, new int[] { 1 }, new int[] { 1 });
                var weights2 = F.Tensors.Slice(weights, new int[] { 2 }, new int[] { 3 }, new int[] { 1 }, new int[] { 1 });
                var weights3 = F.Tensors.Slice(weights, new int[] { 3 }, new int[] { 4 }, new int[] { 1 }, new int[] { 1 });
                var weights4 = F.Tensors.Slice(weights, new int[] { 4 }, new int[] { 5 }, new int[] { 1 }, new int[] { 1 });
                var weights5 = F.Tensors.Slice(weights, new int[] { 5 }, new int[] { 6 }, new int[] { 1 }, new int[] { 1 });
                var weights6 = F.Tensors.Slice(weights, new int[] { 6 }, new int[] { 7 }, new int[] { 1 }, new int[] { 1 });
                var weights7 = F.Tensors.Slice(weights, new int[] { 7 }, new int[] { 8 }, new int[] { 1 }, new int[] { 1 });
                var weights8 = F.Tensors.Slice(weights, new int[] { 8 }, new int[] { 9 }, new int[] { 1 }, new int[] { 1 });
                var weights9 = F.Tensors.Slice(weights, new int[] { 9 }, new int[] { 10 }, new int[] { 1 }, new int[] { 1 });
                var weights10 = F.Tensors.Slice(weights, new int[] { 10 }, new int[] { 11 }, new int[] { 1 }, new int[] { 1 });
                var weights11 = F.Tensors.Slice(weights, new int[] { 11 }, new int[] { 12 }, new int[] { 1 }, new int[] { 1 });
                var weights12 = F.Tensors.Slice(weights, new int[] { 12 }, new int[] { 13 }, new int[] { 1 }, new int[] { 1 });
                var weights13 = F.Tensors.Slice(weights, new int[] { 13 }, new int[] { 14 }, new int[] { 1 }, new int[] { 1 });
                var weights14 = F.Tensors.Slice(weights, new int[] { 14 }, new int[] { 15 }, new int[] { 1 }, new int[] { 1 });
                var weights15 = F.Tensors.Slice(weights, new int[] { 15 }, new int[] { 16 }, new int[] { 1 }, new int[] { 1 });
                var weights16 = F.Tensors.Slice(weights, new int[] { 16 }, new int[] { 17 }, new int[] { 1 }, new int[] { 1 });
                var weights17 = F.Tensors.Slice(weights, new int[] { 17 }, new int[] { 18 }, new int[] { 1 }, new int[] { 1 });
                var weights18 = F.Tensors.Slice(weights, new int[] { 18 }, new int[] { 19 }, new int[] { 1 }, new int[] { 1 });
                var weights19 = F.Tensors.Slice(weights, new int[] { 19 }, new int[] { 20 }, new int[] { 1 }, new int[] { 1 });
                var weights20 = F.Tensors.Slice(weights, new int[] { 20 }, new int[] { 21 }, new int[] { 1 }, new int[] { 1 });
                var weights21 = F.Tensors.Slice(weights, new int[] { 21 }, new int[] { 22 }, new int[] { 1 }, new int[] { 1 });
                var weights22 = F.Tensors.Slice(weights, new int[] { 22 }, new int[] { 23 }, new int[] { 1 }, new int[] { 1 });
                var weights23 = F.Tensors.Slice(weights, new int[] { 23 }, new int[] { 24 }, new int[] { 1 }, new int[] { 1 });
                var weights24 = F.Tensors.Slice(weights, new int[] { 24 }, new int[] { 25 }, new int[] { 1 }, new int[] { 1 });
                var weights25 = F.Tensors.Slice(weights, new int[] { 25 }, new int[] { 26 }, new int[] { 1 }, new int[] { 1 });
                var weights26 = F.Tensors.Slice(weights, new int[] { 26 }, new int[] { 27 }, new int[] { 1 }, new int[] { 1 });
                var weights27 = F.Tensors.Slice(weights, new int[] { 27 }, new int[] { 28 }, new int[] { 1 }, new int[] { 1 });
                var weights28 = F.Tensors.Slice(weights, new int[] { 28 }, new int[] { 29 }, new int[] { 1 }, new int[] { 1 });
                var weights29 = F.Tensors.Slice(weights, new int[] { 29 }, new int[] { 30 }, new int[] { 1 }, new int[] { 1 });
                var weights30 = F.Tensors.Slice(weights, new int[] { 30 }, new int[] { 31 }, new int[] { 1 }, new int[] { 1 });
                var weights31 = F.Tensors.Slice(weights, new int[] { 31 }, new int[] { 32 }, new int[] { 1 }, new int[] { 1 });
                var weights32 = F.Tensors.Slice(weights, new int[] { 32 }, new int[] { 33 }, new int[] { 1 }, new int[] { 1 });
                var weights33 = F.Tensors.Slice(weights, new int[] { 33 }, new int[] { 34 }, new int[] { 1 }, new int[] { 1 });
                var weights34 = F.Tensors.Slice(weights, new int[] { 34 }, new int[] { 35 }, new int[] { 1 }, new int[] { 1 });
                var weights35 = F.Tensors.Slice(weights, new int[] { 35 }, new int[] { 36 }, new int[] { 1 }, new int[] { 1 });
                var weights36 = F.Tensors.Slice(weights, new int[] { 36 }, new int[] { 37 }, new int[] { 1 }, new int[] { 1 });
                var weights37 = F.Tensors.Slice(weights, new int[] { 37 }, new int[] { 38 }, new int[] { 1 }, new int[] { 1 });
                var weights38 = F.Tensors.Slice(weights, new int[] { 38 }, new int[] { 39 }, new int[] { 1 }, new int[] { 1 });
                var weights39 = F.Tensors.Slice(weights, new int[] { 39 }, new int[] { 40 }, new int[] { 1 }, new int[] { 1 });
                var weights40 = F.Tensors.Slice(weights, new int[] { 40 }, new int[] { 41 }, new int[] { 1 }, new int[] { 1 });
                var weights41 = F.Tensors.Slice(weights, new int[] { 41 }, new int[] { 42 }, new int[] { 1 }, new int[] { 1 });
                var weights42 = F.Tensors.Slice(weights, new int[] { 42 }, new int[] { 43 }, new int[] { 1 }, new int[] { 1 });
                var weights43 = F.Tensors.Slice(weights, new int[] { 43 }, new int[] { 44 }, new int[] { 1 }, new int[] { 1 });
                var weights44 = F.Tensors.Slice(weights, new int[] { 44 }, new int[] { 45 }, new int[] { 1 }, new int[] { 1 });
                var weights45 = F.Tensors.Slice(weights, new int[] { 45 }, new int[] { 46 }, new int[] { 1 }, new int[] { 1 });
                var weights46 = F.Tensors.Slice(weights, new int[] { 46 }, new int[] { 47 }, new int[] { 1 }, new int[] { 1 });
                var weights47 = F.Tensors.Slice(weights, new int[] { 47 }, new int[] { 48 }, new int[] { 1 }, new int[] { 1 });
                var weights48 = F.Tensors.Slice(weights, new int[] { 48 }, new int[] { 49 }, new int[] { 1 }, new int[] { 1 });
                var weights49 = F.Tensors.Slice(weights, new int[] { 49 }, new int[] { 50 }, new int[] { 1 }, new int[] { 1 });
                var weights50 = F.Tensors.Slice(weights, new int[] { 50 }, new int[] { 51 }, new int[] { 1 }, new int[] { 1 });
                var weights51 = F.Tensors.Slice(weights, new int[] { 51 }, new int[] { 52 }, new int[] { 1 }, new int[] { 1 });
                var weights52 = F.Tensors.Slice(weights, new int[] { 52 }, new int[] { 53 }, new int[] { 1 }, new int[] { 1 });
                var weights53 = F.Tensors.Slice(weights, new int[] { 53 }, new int[] { 54 }, new int[] { 1 }, new int[] { 1 });
                var weights54 = F.Tensors.Slice(weights, new int[] { 54 }, new int[] { 55 }, new int[] { 1 }, new int[] { 1 });
                var weights55 = F.Tensors.Slice(weights, new int[] { 55 }, new int[] { 56 }, new int[] { 1 }, new int[] { 1 });
                var weights56 = F.Tensors.Slice(weights, new int[] { 56 }, new int[] { 57 }, new int[] { 1 }, new int[] { 1 });
                var weights57 = F.Tensors.Slice(weights, new int[] { 57 }, new int[] { 58 }, new int[] { 1 }, new int[] { 1 });
                var weights58 = F.Tensors.Slice(weights, new int[] { 58 }, new int[] { 59 }, new int[] { 1 }, new int[] { 1 });
                var weights59 = F.Tensors.Slice(weights, new int[] { 59 }, new int[] { 60 }, new int[] { 1 }, new int[] { 1 });
                var weights60 = F.Tensors.Slice(weights, new int[] { 60 }, new int[] { 61 }, new int[] { 1 }, new int[] { 1 });
                var weights61 = F.Tensors.Slice(weights, new int[] { 61 }, new int[] { 62 }, new int[] { 1 }, new int[] { 1 });
                var weights62 = F.Tensors.Slice(weights, new int[] { 62 }, new int[] { 63 }, new int[] { 1 }, new int[] { 1 });
                var weights63 = F.Tensors.Slice(weights, new int[] { 63 }, new int[] { 64 }, new int[] { 1 }, new int[] { 1 });


                var conv0 = F.NN.Conv2D(input0, weights0, newBias, strides.ToArray(), pads, dilation.ToArray(), PadMode.Constant, group);
                var conv1 = F.NN.Conv2D(input1, weights1, newBias, strides.ToArray(), pads, dilation.ToArray(), PadMode.Constant, group);
                var conv2 = F.NN.Conv2D(input2, weights2, newBias, strides.ToArray(), pads, dilation.ToArray(), PadMode.Constant, group);
                var conv3 = F.NN.Conv2D(input3, weights3, newBias, strides.ToArray(), pads, dilation.ToArray(), PadMode.Constant, group);
                var conv4 = F.NN.Conv2D(input4, weights4, newBias, strides.ToArray(), pads, dilation.ToArray(), PadMode.Constant, group);
                var conv5 = F.NN.Conv2D(input5, weights5, newBias, strides.ToArray(), pads, dilation.ToArray(), PadMode.Constant, group);
                var conv6 = F.NN.Conv2D(input6, weights6, newBias, strides.ToArray(), pads, dilation.ToArray(), PadMode.Constant, group);
                var conv7 = F.NN.Conv2D(input7, weights7, newBias, strides.ToArray(), pads, dilation.ToArray(), PadMode.Constant, group);
                var conv8 = F.NN.Conv2D(input8, weights8, newBias, strides.ToArray(), pads, dilation.ToArray(), PadMode.Constant, group);
                var conv9 = F.NN.Conv2D(input9, weights9, newBias, strides.ToArray(), pads, dilation.ToArray(), PadMode.Constant, group);
                var conv10 = F.NN.Conv2D(input10, weights10, newBias, strides.ToArray(), pads, dilation.ToArray(), PadMode.Constant, group);
                var conv11 = F.NN.Conv2D(input11, weights11, newBias, strides.ToArray(), pads, dilation.ToArray(), PadMode.Constant, group);
                var conv12 = F.NN.Conv2D(input12, weights12, newBias, strides.ToArray(), pads, dilation.ToArray(), PadMode.Constant, group);
                var conv13 = F.NN.Conv2D(input13, weights13, newBias, strides.ToArray(), pads, dilation.ToArray(), PadMode.Constant, group);
                var conv14 = F.NN.Conv2D(input14, weights14, newBias, strides.ToArray(), pads, dilation.ToArray(), PadMode.Constant, group);
                var conv15 = F.NN.Conv2D(input15, weights15, newBias, strides.ToArray(), pads, dilation.ToArray(), PadMode.Constant, group);
                var conv16 = F.NN.Conv2D(input16, weights16, newBias, strides.ToArray(), pads, dilation.ToArray(), PadMode.Constant, group);
                var conv17 = F.NN.Conv2D(input17, weights17, newBias, strides.ToArray(), pads, dilation.ToArray(), PadMode.Constant, group);
                var conv18 = F.NN.Conv2D(input18, weights18, newBias, strides.ToArray(), pads, dilation.ToArray(), PadMode.Constant, group);
                var conv19 = F.NN.Conv2D(input19, weights19, newBias, strides.ToArray(), pads, dilation.ToArray(), PadMode.Constant, group);
                var conv20 = F.NN.Conv2D(input20, weights20, newBias, strides.ToArray(), pads, dilation.ToArray(), PadMode.Constant, group);
                var conv21 = F.NN.Conv2D(input21, weights21, newBias, strides.ToArray(), pads, dilation.ToArray(), PadMode.Constant, group);
                var conv22 = F.NN.Conv2D(input22, weights22, newBias, strides.ToArray(), pads, dilation.ToArray(), PadMode.Constant, group);
                var conv23 = F.NN.Conv2D(input23, weights23, newBias, strides.ToArray(), pads, dilation.ToArray(), PadMode.Constant, group);
                var conv24 = F.NN.Conv2D(input24, weights24, newBias, strides.ToArray(), pads, dilation.ToArray(), PadMode.Constant, group);
                var conv25 = F.NN.Conv2D(input25, weights25, newBias, strides.ToArray(), pads, dilation.ToArray(), PadMode.Constant, group);
                var conv26 = F.NN.Conv2D(input26, weights26, newBias, strides.ToArray(), pads, dilation.ToArray(), PadMode.Constant, group);
                var conv27 = F.NN.Conv2D(input27, weights27, newBias, strides.ToArray(), pads, dilation.ToArray(), PadMode.Constant, group);
                var conv28 = F.NN.Conv2D(input28, weights28, newBias, strides.ToArray(), pads, dilation.ToArray(), PadMode.Constant, group);
                var conv29 = F.NN.Conv2D(input29, weights29, newBias, strides.ToArray(), pads, dilation.ToArray(), PadMode.Constant, group);
                var conv30 = F.NN.Conv2D(input30, weights30, newBias, strides.ToArray(), pads, dilation.ToArray(), PadMode.Constant, group);
                var conv31 = F.NN.Conv2D(input31, weights31, newBias, strides.ToArray(), pads, dilation.ToArray(), PadMode.Constant, group);
                var conv32 = F.NN.Conv2D(input32, weights32, newBias, strides.ToArray(), pads, dilation.ToArray(), PadMode.Constant, group);
                var conv33 = F.NN.Conv2D(input33, weights33, newBias, strides.ToArray(), pads, dilation.ToArray(), PadMode.Constant, group);
                var conv34 = F.NN.Conv2D(input34, weights34, newBias, strides.ToArray(), pads, dilation.ToArray(), PadMode.Constant, group);
                var conv35 = F.NN.Conv2D(input35, weights35, newBias, strides.ToArray(), pads, dilation.ToArray(), PadMode.Constant, group);
                var conv36 = F.NN.Conv2D(input36, weights36, newBias, strides.ToArray(), pads, dilation.ToArray(), PadMode.Constant, group);
                var conv37 = F.NN.Conv2D(input37, weights37, newBias, strides.ToArray(), pads, dilation.ToArray(), PadMode.Constant, group);
                var conv38 = F.NN.Conv2D(input38, weights38, newBias, strides.ToArray(), pads, dilation.ToArray(), PadMode.Constant, group);
                var conv39 = F.NN.Conv2D(input39, weights39, newBias, strides.ToArray(), pads, dilation.ToArray(), PadMode.Constant, group);
                var conv40 = F.NN.Conv2D(input40, weights40, newBias, strides.ToArray(), pads, dilation.ToArray(), PadMode.Constant, group);
                var conv41 = F.NN.Conv2D(input41, weights41, newBias, strides.ToArray(), pads, dilation.ToArray(), PadMode.Constant, group);
                var conv42 = F.NN.Conv2D(input42, weights42, newBias, strides.ToArray(), pads, dilation.ToArray(), PadMode.Constant, group);
                var conv43 = F.NN.Conv2D(input43, weights43, newBias, strides.ToArray(), pads, dilation.ToArray(), PadMode.Constant, group);
                var conv44 = F.NN.Conv2D(input44, weights44, newBias, strides.ToArray(), pads, dilation.ToArray(), PadMode.Constant, group);
                var conv45 = F.NN.Conv2D(input45, weights45, newBias, strides.ToArray(), pads, dilation.ToArray(), PadMode.Constant, group);
                var conv46 = F.NN.Conv2D(input46, weights46, newBias, strides.ToArray(), pads, dilation.ToArray(), PadMode.Constant, group);
                var conv47 = F.NN.Conv2D(input47, weights47, newBias, strides.ToArray(), pads, dilation.ToArray(), PadMode.Constant, group);
                var conv48 = F.NN.Conv2D(input48, weights48, newBias, strides.ToArray(), pads, dilation.ToArray(), PadMode.Constant, group);
                var conv49 = F.NN.Conv2D(input49, weights49, newBias, strides.ToArray(), pads, dilation.ToArray(), PadMode.Constant, group);
                var conv50 = F.NN.Conv2D(input50, weights50, newBias, strides.ToArray(), pads, dilation.ToArray(), PadMode.Constant, group);
                var conv51 = F.NN.Conv2D(input51, weights51, newBias, strides.ToArray(), pads, dilation.ToArray(), PadMode.Constant, group);
                var conv52 = F.NN.Conv2D(input52, weights52, newBias, strides.ToArray(), pads, dilation.ToArray(), PadMode.Constant, group);
                var conv53 = F.NN.Conv2D(input53, weights53, newBias, strides.ToArray(), pads, dilation.ToArray(), PadMode.Constant, group);
                var conv54 = F.NN.Conv2D(input54, weights54, newBias, strides.ToArray(), pads, dilation.ToArray(), PadMode.Constant, group);
                var conv55 = F.NN.Conv2D(input55, weights55, newBias, strides.ToArray(), pads, dilation.ToArray(), PadMode.Constant, group);
                var conv56 = F.NN.Conv2D(input56, weights56, newBias, strides.ToArray(), pads, dilation.ToArray(), PadMode.Constant, group);
                var conv57 = F.NN.Conv2D(input57, weights57, newBias, strides.ToArray(), pads, dilation.ToArray(), PadMode.Constant, group);
                var conv58 = F.NN.Conv2D(input58, weights58, newBias, strides.ToArray(), pads, dilation.ToArray(), PadMode.Constant, group);
                var conv59 = F.NN.Conv2D(input59, weights59, newBias, strides.ToArray(), pads, dilation.ToArray(), PadMode.Constant, group);
                var conv60 = F.NN.Conv2D(input60, weights60, newBias, strides.ToArray(), pads, dilation.ToArray(), PadMode.Constant, group);
                var conv61 = F.NN.Conv2D(input61, weights61, newBias, strides.ToArray(), pads, dilation.ToArray(), PadMode.Constant, group);
                var conv62 = F.NN.Conv2D(input62, weights62, newBias, strides.ToArray(), pads, dilation.ToArray(), PadMode.Constant, group);
                var conv63 = F.NN.Conv2D(input63, weights63, newBias, strides.ToArray(), pads, dilation.ToArray(), PadMode.Constant, group);

                var add_0 = F.Math.Add(conv0, conv1);
                var add_1 = F.Math.Add(add_0, conv2);
                var add_2 = F.Math.Add(add_1, conv3);
                var add_3 = F.Math.Add(add_2, conv4);
                var add_4 = F.Math.Add(add_3, conv5);
                var add_5 = F.Math.Add(add_4, conv6);
                var add_6 = F.Math.Add(add_5, conv7);
                var add_7 = F.Math.Add(add_6, conv8);
                var add_8 = F.Math.Add(add_7, conv9);
                var add_9 = F.Math.Add(add_8, conv10);
                var add_10 = F.Math.Add(add_9, conv11);
                var add_11 = F.Math.Add(add_10, conv12);
                var add_12 = F.Math.Add(add_11, conv13);
                var add_13 = F.Math.Add(add_12, conv14);
                var add_14 = F.Math.Add(add_13, conv15);
                var add_15 = F.Math.Add(add_14, conv16);
                var add_16 = F.Math.Add(add_15, conv17);
                var add_17 = F.Math.Add(add_16, conv18);
                var add_18 = F.Math.Add(add_17, conv19);
                var add_19 = F.Math.Add(add_18, conv20);
                var add_20 = F.Math.Add(add_19, conv21);
                var add_21 = F.Math.Add(add_20, conv22);
                var add_22 = F.Math.Add(add_21, conv23);
                var add_23 = F.Math.Add(add_22, conv24);
                var add_24 = F.Math.Add(add_23, conv25);
                var add_25 = F.Math.Add(add_24, conv26);
                var add_26 = F.Math.Add(add_25, conv27);
                var add_27 = F.Math.Add(add_26, conv28);
                var add_28 = F.Math.Add(add_27, conv29);
                var add_29 = F.Math.Add(add_28, conv30);
                var add_30 = F.Math.Add(add_29, conv31);
                var add_31 = F.Math.Add(add_30, conv32);
                var add_32 = F.Math.Add(add_31, conv33);
                var add_33 = F.Math.Add(add_32, conv34);
                var add_34 = F.Math.Add(add_33, conv35);
                var add_35 = F.Math.Add(add_34, conv36);
                var add_36 = F.Math.Add(add_35, conv37);
                var add_37 = F.Math.Add(add_36, conv38);
                var add_38 = F.Math.Add(add_37, conv39);
                var add_39 = F.Math.Add(add_38, conv40);
                var add_40 = F.Math.Add(add_39, conv41);
                var add_41 = F.Math.Add(add_40, conv42);
                var add_42 = F.Math.Add(add_41, conv43);
                var add_43 = F.Math.Add(add_42, conv44);
                var add_44 = F.Math.Add(add_43, conv45);
                var add_45 = F.Math.Add(add_44, conv46);
                var add_46 = F.Math.Add(add_45, conv47);
                var add_47 = F.Math.Add(add_46, conv48);
                var add_48 = F.Math.Add(add_47, conv49);
                var add_49 = F.Math.Add(add_48, conv50);
                var add_50 = F.Math.Add(add_49, conv51);
                var add_51 = F.Math.Add(add_50, conv52);
                var add_52 = F.Math.Add(add_51, conv53);
                var add_53 = F.Math.Add(add_52, conv54);
                var add_54 = F.Math.Add(add_53, conv55);
                var add_55 = F.Math.Add(add_54, conv56);
                var add_56 = F.Math.Add(add_55, conv57);
                var add_57 = F.Math.Add(add_56, conv58);
                var add_58 = F.Math.Add(add_57, conv59);
                var add_59 = F.Math.Add(add_58, conv60);
                var add_60 = F.Math.Add(add_59, conv61);
                var add_61 = F.Math.Add(add_60, conv62);
                var add_62 = F.Math.Add(add_61, conv63);
                var conv = add_62 + bias; // Final result

                List<string> outputNames = new() { op.Output[0] };
                conv.Metadata.OutputNames = outputNames;

                return conv;

            }
            else
            {
                var conv = F.NN.Conv2D(input, weights, bias, strides.ToArray(), pads, dilation.ToArray(), PadMode.Constant, group);
                List<string> outputNames = new() { op.Output[0] };
                conv.Metadata.OutputNames = outputNames;
                if (isConv1D)
                {
                    conv = Squeeze(conv, new[] { 2 });
                }

                return conv;
            }
        }

        private Call To4D(Expr input) => Unsqueeze(input, new[] { 2 });

        private bool IsConv1D(Expr weights)
        {
            bool conv1d = false;
            weights.InferenceType();
            var weightsRank = weights.CheckedShape.Rank;
            switch (weightsRank)
            {
                case 3:
                    conv1d = true;
                    break;
                case 4:
                    break;
                default:
                    throw new NotSupportedException($"only support 1d and 2d, but get weights rank {weightsRank}");
            }

            return conv1d;
        }

        private Expr GetPadsAttribute(NodeProto op, bool isConv1D = false)
        {
            var paddings = GetIntsAttribute(op, "pads", 0, 4);
            if (isConv1D)
            {
                paddings = new[] { 0, paddings[0], 0, paddings[1] };
            }

            return ToNncasePadFormat(paddings);
        }

        private Tensor GetStrideAttribute(NodeProto op)
        {
            return Tensor.From<long>(GetIntsAttribute(op, "strides", 1, 2));
        }

        private long[] GetDilationsAttribute(NodeProto op)
        {
            return GetIntsAttribute(op, "dilations", new[] { 1, 1 });
        }

        private Expr GetBias(NodeProto op, Expr weights, bool isConvTranspose = false, long groups = 1)
        {
            var biasSizeIndex = isConvTranspose ? 1 : 0;
            return op.Input.Count > 2
                ? GetInputExpr(op, 2)
                : F.Tensors.Expand(0f, Util.ShapeIndex(weights, biasSizeIndex) * groups);
        }

        private Expr AutoPad(NodeProto op, string autoPad, Expr input, Expr weights, long[] strides, long[] dilation, bool isConv1D = false) => autoPad switch
        {
            "NOTSET" => GetPadsAttribute(op, isConv1D),
            "SAME_UPPER" => Util.GetPaddings(input, weights, strides, dilation, true),
            "SAME_LOWER" => Util.GetPaddings(input, weights, strides, dilation, true, true),
            "VALID" => GetPadsAttribute(op, isConv1D),

            // when VALID, I'm not sure this is correct
            // in onnx doc, not spec when VALID value
            // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Conv
            _ => throw new InvalidDataException($"invalid AutoPad Value: {autoPad}"),
        };
    }
}
