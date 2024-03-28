// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NetFabric.Hyperlinq;
using Nncase.ArgsStruct;
using Nncase.Evaluator;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.Ncnn;
using Nncase.PatternMatch;
using static Nncase.IR.F.Ncnn;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.NN;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Ncnn;

[RuleGenerator]
public partial class LowerConvTranspose : RewriteRule<Pattern>
{
    /// <inheritdoc/>
    public override Pattern Pattern { get; } = IsConv2DTranspose(
    "conv",
    conv => conv.PadMode == PadMode.Constant,
    IsWildcard("input"),
    IsTensorConst("weights"),
    IsTensorConst("bias"),
    IsTensorConst("outputShape"),
    IsTensorConst("strides"),
    IsTensorConst("paddings"),
    IsTensorConst("outputPadding"),
    IsTensorConst("dilation"),
    IsTensorConst("group"),
    IsTensorConst("fusedClamp"));

    private Expr? GetReplace(Expr input, Tensor<float> weights, Tensor<float> bias, int[] outputShape, int[] strides, int[] paddings, int[] outputPadding, int[] dilation, int group, float[] fusedClamp)
    {
        if (input.CheckedShape.Rank != 4 || input.CheckedShape[0].FixedValue != 1)
        {
            return null;
        }

        int[] weightShape = weights.Shape.ToValueArray();
        var (numOutput, kernelH, kernelW) = (weights.Shape[0], weights.Shape[2], weights.Shape[3]);
        var (dilationH, dilationW) = (dilation[0], dilation.Length == 2 ? dilation[1] : dilation[0]);
        var (strideH, strideW) = (strides[0], strides.Length == 2 ? strides[1] : strides[0]);

        var (padLeft, padRight, padTop, padBottom) = (paddings[2], paddings[3], paddings[0], paddings[1]);
        int biasTerm = 1;
        var weightsDataSize = weightShape[0] * weightShape[1] * weightShape[2] * weightShape[3];

        // 1:Relu 2: leaky relu 3:clip 4: sigmoid 5: mish 6: hardswish
        int activationType = 3;

        // Note: Has reordered in importer.
        // reorder weights(Cin Cout H W) to (Cout Cin H W)
        // var newWeights = Transpose(weights, new int[] { 1, 0, 2, 3 }).Evaluate().AsTensor();

        // actType and actParams not used in ncnn: onnx2ncnn.
        var args = new ConvTransposeArgs(weights, bias.ToArray(), numOutput.FixedValue, kernelW.FixedValue, kernelH.FixedValue, dilationW, dilationH, strideW, strideH, padLeft, padRight, padTop, padBottom, biasTerm, weightsDataSize, activationType, fusedClamp, outputPadding[1], outputPadding[0], outputShape[3], outputShape[2]);

        var inRes = Squeeze(input, new[] { 0 });
        var inResO = new Var(inRes.CheckedType);
        var conv = new Call(new Fusion("ncnn", NcnnConvTranspose(inResO, args), new[] { inResO }), inRes);

        var outRes = Unsqueeze(conv, new[] { 0 });

        return outRes;
    }
}
