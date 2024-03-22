// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NetFabric.Hyperlinq;
using Nncase.ArgsStruct;
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
public partial class LowerConv : RewriteRule<Pattern>
{
    /// <inheritdoc/>
    public override Pattern Pattern { get; } = IsConv2D(
    "conv",
    conv => conv.PadMode == PadMode.Constant,
    IsWildcard("input"),
    IsTensorConst("weights"),
    IsTensorConst("bias"),
    IsTensorConst("strides"),
    IsTensorConst("paddings"),
    IsTensorConst("dilation"),
    IsTensorConst("groups"),
    IsTensorConst("fusedClamp"));

    private Expr? GetReplace(Expr input, Tensor<float> weights, Tensor<float> bias, int[] strides, int[] paddings, int[] dilation, int groups, float[] fusedClamp)
    {
        var (numOutput, kernelH, kernelW) = (weights.Shape[0], weights.Shape[2], weights.Shape[3]);
        var (dilationH, dilationW) = (dilation[0], dilation.Length == 2 ? dilation[1] : dilation[0]);
        var (strideH, strideW) = (strides[0], strides.Length == 2 ? strides[1] : strides[0]);

        // int[] paddingsValue = ((TensorConst)paddings).Value.ToArray<int>();
        var (padLeft, padRight, padTop, padBottom) = (paddings[2], paddings[3], paddings[0], paddings[1]);
        float padValue = 0;
        int biasTerm = 1;
        var weightsDataSize = weights.Shape[0] * weights.Shape[1] * weights.Shape[2] * weights.Shape[3];
        int int8ScaleTerm = 0;

        // 1:Relu 2: leaky relu 3:clip 4: sigmoid 5: mish 6: hardswish
        int activationType = 3;
        int dynamicWeight = 0;

        var args = new ConvArgs(
            weights.ToArray(),
            bias.ToArray(),
            numOutput.FixedValue,
            kernelW.FixedValue,
            kernelH.FixedValue,
            dilationW,
            dilationH,
            strideW,
            strideH,
            padLeft,
            padRight,
            padTop,
            padBottom,
            padValue,
            biasTerm,
            weightsDataSize.FixedValue,
            int8ScaleTerm,
            activationType,
            fusedClamp,
            dynamicWeight,
            groups);

        var inRes = Squeeze(input, new[] { 0 });
        var inResO = new Var(inRes.CheckedType);
        var conv = new Call(new Fusion("ncnn", NcnnConv(inResO, args), new[] { inResO }), inRes);

        var outRes = Unsqueeze(conv, new[] { 0 });

        return outRes;
    }
}
