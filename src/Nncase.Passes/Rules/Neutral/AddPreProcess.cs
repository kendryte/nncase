// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Toolkit.HighPerformance;
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
    /// <summary>
    /// Main func for AddPreProcess.
    /// </summary>
    /// <param name="module"> The graph. </param>
    /// <param name="options"> RunPassContext. </param>
    /// <returns> Return a new graph with preprocess and postprocess. </returns>
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

        foreach (var input in entry.Parameters)
        {
            var a = new Var(new TensorType(newType[(int)inputType], inputShape));

            Expr newInput = a;
            var oldShape = input.CheckedShape;

            // Convert new input to NCHW
            var newInputPerm = Array.Empty<int>();
            if (inputLayout != string.Empty)
            {
                if (inputLayout != "NHWC" && inputLayout != "NCHW")
                {
                    newInputPerm = Array.ConvertAll(
                        inputLayout.Replace(" ", string.Empty, StringComparison.OrdinalIgnoreCase).Split(","),
                        int.Parse);
                }

                newInput = inputLayout switch
                {
                    "NHWC" => Transpose(newInput, new[] {0, 3, 1, 2}),
                    "NCHW" => Transpose(newInput, new[] { 0, 1, 2, 3 }),
                    _ => Transpose(newInput, newInputPerm),
                };
            }

            int n = 0, c = 0, h = 0, w = 0;
            if (inputShape.Length == 4)
            {
                if (inputLayout == "NHWC")
                {
                    (n, h, w, c) = (inputShape[0], inputShape[1], inputShape[2], inputShape[3]);
                }
                else if (inputLayout == "NCHW")
                {
                    (n, c, h, w) = (inputShape[0], inputShape[1], inputShape[2], inputShape[3]);
                }
                else
                {
                    (n, c, h, w) = (inputShape[newInputPerm[0]], inputShape[newInputPerm[1]], inputShape[newInputPerm[2]], inputShape[newInputPerm[3]]);
                }

                // SwapRB
                if (swapRB && c != 1)
                {
                    var axes = new int[4] {0, 1, 2, 3};
                    var strides = new int[4] {1, 1, 1, 1};
                    newInput = Concat(
                        new IR.Tuple(new[]
                        {
                            Slice(newInput, new int[4] {0, 2, 0, 0}, new int[4] {n, 3, h, w}, axes, strides),
                            Slice(newInput, new int[4] {0, 1, 0, 0}, new int[4] {n, 2, h, w}, axes, strides),
                            Slice(newInput, new int[4] {0, 0, 0, 0}, new int[4] {n, 1, h, w}, axes, strides),
                        }),
                        1);

                    // TODO: fix slice neg strides shape inference
                    // newInput = Slice(newInput, new int[] {n, c, h, w },new[] { 0, 0, 0, 0 },  axes, strides);
                }
            }

            // Dequantize to float
            if (inputType != InputType.Float32)
            {
                var qP = QuantParamOf(QuantMode.UnsignedMode, new[] { inputRange[0], inputRange[1] }, 8);
                newInput = Dequantize(newInput, qP, DataTypes.Float32);
            }

            // Letterbox
            if (inputShape.Length == 4)
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

                if (modelH != h || modelW != w)
                {
                    var ratio = Math.Min(modelH / (float)h, modelW / (float)w);

                    var pads = Tensor.From<int>(new[] {0, 0, 0, 0, 0, 0, 0, 0}, new Shape(new[] {4, 2}));

                    var resizeH = Math.Round(h * ratio);
                    var resizeW = Math.Round(w * ratio);

                    var padH = modelH - resizeH;
                    var padW = modelW - resizeW;
                    var resizeShape = new int[] {n, c, (int)resizeH, (int)resizeW};

                    pads[2, 0] = (int) Math.Round((padH / 2) - 0.1);
                    pads[2, 1] = (int) padH - (int) Math.Round((padH / 2) - 0.1);
                    pads[3, 0] = (int) Math.Round((padW / 2) - 0.1);
                    pads[3, 1] = (int) padW - (int) Math.Round((padW / 2) - 0.1);

                    newInput = IR.F.NN.Pad(
                        IR.F.Imaging.ResizeImage(ImageResizeMode.Bilinear, newInput, float.NaN, resizeShape,
                            ImageResizeTransformationMode.HalfPixel), pads, PadMode.Constant, letterBoxValue);
                }
            }

            // Normalization
            if (mean.Length != 0)
            {
                newInput = mean.Length switch
                {
                    3 when inputShape.Length == 4 => (newInput - Tensor.From(mean, new[] { 1, mean.Length, 1, 1 })) /
                                                     Tensor.From(std, new[] { 1, std.Length, 1, 1 }),
                    _ => (newInput - Tensor.From(new float[] { mean[0] }, new[] { 1 })) /
                         Tensor.From(new float[] { std[0] }, new[] { 1 }),
                };

                // newInput = Binary(BinaryOp.Div, Binary(BinaryOp.Sub, newInput, Tensor.From(mean, new []{1,3,1,1})), Const.FromTensor(std) );
            }

            // Convert to model layout
            if (modelLayout == "NHWC" && inputShape.Length == 4)
            {
                newInput = Transpose(newInput, new[] {0, 2, 3, 1});
            }

            new Passes.Mutators.Substitutor(expr => object.ReferenceEquals(expr, input) ? newInput : null).Rewrite(
                entry.Body);
            new Passes.Mutators.Substitutor(expr => object.ReferenceEquals(expr, input) ? a : null).Rewrite(entry);
        }

        return Task.FromResult(module);
    }
}
