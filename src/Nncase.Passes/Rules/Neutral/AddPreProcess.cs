// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.IR.Imaging;
using Nncase.IR.Math;
using Nncase.Passes;
using Nncase.PatternMatch;
using OrtKISharp;
using static Nncase.IR.F.Math;
using static Nncase.IR.F.NN;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.Utility;
using Pad = Nncase.IR.NN.Pad;

namespace Nncase.Passes.Rules.Neutral;

/// <summary>
/// Add preprocess in model.
/// </summary>
[RuleGenerator]
public sealed class AddPreProcess : ModulePass
{
    protected override Task<IRModule> RunCoreAsync(IRModule module, RunPassContext options)
    {
        var preProcess = CompileSession.CompileOptions.PreProcess;
        var inputLayout = CompileSession.CompileOptions.InputLayout;
        var inputType = CompileSession.CompileOptions.InputType;
        var inputShape = CompileSession.CompileOptions.InputShape;
        var inputRange = CompileSession.CompileOptions.InputRange;
        var swapRB = CompileSession.CompileOptions.SwapRB;
        var letterBoxValue = CompileSession.CompileOptions.LetterBoxValue;
        var mean = CompileSession.CompileOptions.Mean;
        var std = CompileSession.CompileOptions.Std;
        var modelLayout = CompileSession.CompileOptions.ModelLayout;

        var entry = (IR.Function)module.Entry!;
        var newType = new[] { DataTypes.UInt8, DataTypes.Int8, DataTypes.Float32 };

        if (!preProcess)
        {
            return Task.FromResult(module);
        }

        var a = new Var(new TensorType(newType[(int)inputType], inputShape));
        foreach (var input in entry.Parameters)
        {
            Expr newInput = a;
            var oldShape = input.CheckedShape;

            int n, c, h, w;
            if (inputLayout == "NHWC")
            {
                (n, h, w, c) = (inputShape[0], inputShape[1], inputShape[2], inputShape[3]);
            }
            else
            {
                (n, c, h, w) = (inputShape[0], inputShape[1], inputShape[2], inputShape[3]);
            }

            // Convert new input to NCHW
            if (inputLayout == "NHWC")
            {
                newInput = Transpose(newInput, new int[4] { 0, 3, 1, 2 });
            }

            // SwapRB
            if (swapRB)
            {
                var axes = new int[4] { 0, 1, 2, 3 };
                var strides = new int[4] { 1, 1, 1, 1 };
                newInput = Concat(
                new IR.Tuple(new[] { Slice(newInput, new int[4] { 0, 2, 0, 0 }, new int[4] { n, 3, h, w }, axes, strides),
                                     Slice(newInput, new int[4] { 0, 1, 0, 0 }, new int[4] { n, 2, h, w }, axes, strides),
                                     Slice(newInput, new int[4] { 0, 0, 0, 0 }, new int[4] { n, 1, h, w }, axes, strides), }),
                1);

                // TODO: fix slice neg strides shape inference
                // newInput = Slice(newInput, new int[] {n, c, h, w },new[] { 0, 0, 0, 0 },  axes, strides);
            }

            // Dequantize to float
            if (inputType != InputType.Float32)
            {
                var qP = QuantParamOf(QuantMode.UnsignedMode, new[] { inputRange[0], inputRange[1] }, 8);
                newInput = Dequantize(newInput, qP, DataTypes.Float32);
            }

            // Letterbox
            {
                int modelH, modelW;

                if (modelLayout != "NCHW")
                {
                    (modelH, modelW) = (oldShape[1].FixedValue, oldShape[2].FixedValue);
                }
                else
                {
                    (modelH, modelW) = (oldShape[2].FixedValue, oldShape[3].FixedValue);
                }

                var ratio = Math.Min(modelH / (float)h, modelW / (float)w);

                var pads = Tensor.From<int>(new[] { 0, 0, 0, 0, 0, 0, 0, 0 }, new Shape(new[] { 4, 2 }));

                var resizeH = Math.Round(h * ratio);
                var resizeW = Math.Round(w * ratio);

                var padH = modelH - resizeH;
                var padW = modelW - resizeW;
                var resizeShape = new int[] { n, c, (int)resizeH, (int)resizeW };

                pads[2, 0] = (int)Math.Round((padH / 2) - 0.1);
                pads[2, 1] = (int)padH - (int)Math.Round((padH / 2) - 0.1);
                pads[3, 0] = (int)Math.Round((padW / 2) - 0.1);
                pads[3, 1] = (int)padW - (int)Math.Round((padW / 2) - 0.1);

                newInput = IR.F.NN.Pad(IR.F.Imaging.ResizeImage(ImageResizeMode.Bilinear, newInput, float.NaN, resizeShape, ImageResizeTransformationMode.HalfPixel), pads, PadMode.Constant, letterBoxValue);
            }

            // Normalization
            if (mean.Length != 0)
            {
                newInput = (newInput - Tensor.From(mean, new[] { 1, 3, 1, 1 })) / Tensor.From(std, new[] { 1, 3, 1, 1 });

                // newInput = Binary(BinaryOp.Div, Binary(BinaryOp.Sub, newInput, Tensor.From(mean, new []{1,3,1,1})), Const.FromTensor(std) );
            }

            // Convert to model layout
            if (modelLayout == "NHWC")
            {
                newInput = Transpose(newInput, new[] { 0, 2, 3, 1 });
            }

            var y = new Passes.Mutators.Substitutor(expr => object.ReferenceEquals(expr, input) ? newInput : null).Rewrite(entry.Body);
            var x = (Function)new Passes.Mutators.Substitutor(expr => object.ReferenceEquals(expr, input) ? a : null).Rewrite(entry);
        }

        return Task.FromResult(module);
    }
}
