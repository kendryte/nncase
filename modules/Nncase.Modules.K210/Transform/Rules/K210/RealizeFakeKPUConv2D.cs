// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

#if false
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.Evaluator;
using Nncase.IR;
using Nncase.IR.F;
using Nncase.IR.K210;
using Nncase.IR.Math;
using static Nncase.PatternMatch.F.K210;
using Nncase.PatternMatch;
using Nncase.Utilities;
using Tensorflow.Keras;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.F.NN;
using static Nncase.PatternMatch.Utility;
using Math = System.Math;

namespace Nncase.Passes.Rules.K210;

/// <summary>
/// Lower <see cref="IR.K210.FakeKPUConv2D"/> to <see cref="IR.K210.KPUConv2D"/>.
/// </summary>
[RuleGenerator]
public sealed partial class RealizeFakeKPUConv2D : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } = IsFakeKPUConv2D(
            null,
            "fake_conv2d_call",
            op => true,
            IsRangeOfMarker(IsWildcard("input")
                    with
            { TypePattern = HasFixedShape() },
                IsConst("input_range")),
            IsTensorConst("weights")
            //IsTensorConst()
            // IsTensorConst("bias")
            // IsTensorConst("batchNorms"),
            // IsTensorConst("activation")
            );

    private Expr? GetReplace(Call fake_conv2d_call, Expr input, Tensor<float> input_range, Expr weights)
    {
        var inDType = input.CheckedDataType;
        var inShape = input.CheckedShape;
        var wShape = weights.CheckedShape;
        var outShape = fake_conv2d_call.CheckedShape;
        var inChannels = inShape[1].FixedValue;
        var outChannels = outShape[1].FixedValue;
        var filterH = wShape[2].FixedValue;
        var filterW = wShape[3].FixedValue;

        var groups = 1;
        if ((groups == 1 || groups == inChannels)
            && KPUUtility.IsSupportedShape(inShape)
            && KPUUtility.IsSupportedShape(outShape)
            && KPUUtility.TryGetFilterType(filterH, filterW, out var filterType))
        {
            var isDepthwise = inChannels == outChannels && outChannels == groups;
            var batchnorms = None.Default;
            var act = None.Default;
            //FakeKPUConv2D conv = null;
            //var bias = conv.Bias.ToScalar();
            //var bias = (TensorConst)IR.F.Random.Normal(DataTypes.Float32, new[] {16}).Evaluate().AsTensor();
            //var act = KPUUtility.GetDefaultConvActParam(weights, bias);
            return new Call(new IR.K210.KPUConv2D(isDepthwise, filterType, KPUPoolType.Bypass, KPUUtility.Activation()), input, weights, batchnorms, act);
            // return new Function(IR.F.K210.KPUConv2D(isDepthwise, filterType, KPUPoolType.Bypass, KPUUtility.Activation(), input,
            //     weights, batchnorms, act));
        }

        return null;
    }

    /*public static KPUActivationParameters quantizeAct(Quantize quantize, float actScale, QuantParam yQuantParam, QuantParam zQuantParam, ValueRange<float> activation )
    {
        // var xfMin = IR.F.Math.Clamp((0 - yQuantParam.ZeroPoint) * yQuantParam.Scale, activation.Min, activation.Max);
        // var xfMax = IR.F.Math.Clamp((255 - yQuantParam.ZeroPoint) * yQuantParam.Scale, activation.Min, activation.Max);
        var xfMin = Math.Min(Math.Max((0 - yQuantParam.ZeroPoint) * yQuantParam.Scale, activation.Min), activation.Max);
        var xfMax = Math.Min(Math.Max((255 - yQuantParam.ZeroPoint) * yQuantParam.Scale, activation.Min), activation.Max);

        var zqScale = actScale / zQuantParam.Scale;

        var samplesCount = 2048;
        var sampleStep = (xfMax - xfMin) / (samplesCount - 1);
        float[] samplesX = new float[samplesCount];
        float[] samplesY = new float[samplesCount];
        for (var i = 0; i < samplesCount; i++)
        {
            samplesX[i] = xfMin + i * sampleStep;
            samplesY[i] = xfMin + i * sampleStep;
        }

        KPUActivationParameters act = new KPUActivationParameters();
        return act;
    }

    public static KPUBatchNormParameters quantizeBn()
    {
        KPUBatchNormParameters bn = new KPUBatchNormParameters();
        return bn;
    }*/
}
#endif
