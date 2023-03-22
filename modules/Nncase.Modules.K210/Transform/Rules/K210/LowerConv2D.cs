// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.IR.F;
using Nncase.IR.K210;
using Nncase.PatternMatch;
using Nncase.Utilities;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.NN;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.K210;

/// <summary>
/// Lower <see cref="IR.NN.Conv2D"/> to <see cref="IR.K210.FakeKPUConv2D"/>.
/// </summary>
[RuleGenerator]
public sealed partial class LowerConv2D : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } = IsConv2D(
        null,
        "conv2d",
        PadMode.Constant,
        IsWildcard("input") with { TypePattern = HasFixedShape() },
        IsTensorConst("weights"),
        IsTensorConst("bias"),
        IsTensorConst("strides"),
        IsTensorConst("paddings"),
        new[] { 1, 1 },
        IsTensorConst("groups"),
        IsTensorConst("fusedClamp")) with
    {
        TypePattern = HasFixedShape(),
    };

    private Expr? GetReplace(Expr conv2d, Expr input, Expr weights, Tensor<float> bias, int[] strides, Tensor<int> paddings, int groups, float[] fusedClamp)
    {
        var inDType = input.CheckedDataType;
        var inShape = input.CheckedShape;
        var wShape = weights.CheckedShape;
        var outShape = conv2d.CheckedShape;
        var inChannels = inShape[1].FixedValue;
        var outChannels = outShape[1].FixedValue;
        var filterH = wShape[2].FixedValue;
        var filterW = wShape[3].FixedValue;

        if ((groups == 1 || groups == inChannels)
            && KPUUtility.IsSupportedShape(inShape)
            && KPUUtility.IsSupportedShape(outShape)
            && KPUUtility.TryGetFilterType(filterH, filterW, out var filterType))
        {
            var zeroOfDType = ((Tensor)0f).CastTo(inDType);
            var zeroPaddings = new[] { 0, 0 };
            var isDepthwise = inChannels == outChannels && outChannels == groups;
            var kpuPad = KPUUtility.GetKPUPadding(filterType);
            var padH = new[] { paddings[0, 0] - kpuPad, paddings[0, 1] - kpuPad };
            var padW = new[] { paddings[1, 0] - kpuPad, paddings[1, 1] - kpuPad };

            var prePaddings = new[] { zeroPaddings, zeroPaddings, KPUUtility.GetPrePadding(padH), KPUUtility.GetPrePadding(padW) }.To2D();
            var prePad = NN.Pad(input, prePaddings, PadMode.Constant, zeroOfDType);
            CompilerServices.InferenceType(prePad);
            if (KPUUtility.IsSupportedShape(prePad.CheckedShape))
            {
                var inQuantParam = IR.F.Math.QuantParamOf(QuantMode.UnsignedMode, IR.F.Math.RangeOf(prePad), 8);
                var quant = IR.F.Math.FakeQuantize(prePad, inQuantParam, DataTypes.UInt8);

                var kpuUpload = IR.F.K210.FakeKPUUpload(quant);

                var inRangeMarker = IR.F.Math.RangeOfMarker(kpuUpload, IR.F.Math.RangeOf(kpuUpload));
                var kpuConv = IR.F.K210.FakeKPUConv2D(isDepthwise, filterType, KPUPoolType.Bypass, bias, (fusedClamp[0], fusedClamp[1]), inRangeMarker, weights);
                var outRangeMarker = IR.F.Math.RangeOfMarker(kpuConv, IR.F.Math.RangeOf(kpuConv));

                var kpuDownload = IR.F.K210.FakeKPUDownload(outRangeMarker);

                var outQuantParam = IR.F.Math.QuantParamOf(QuantMode.UnsignedMode, IR.F.Math.RangeOf(kpuDownload), 8);
                var dequant = IR.F.Math.FakeDequantize(kpuDownload, outQuantParam, inDType);
                var postPaddings = new[] { zeroPaddings, zeroPaddings, KPUUtility.GetPostPadding(padH), KPUUtility.GetPostPadding(padW) }.To2D();
                var postPad = NN.Pad(dequant, postPaddings, PadMode.Constant, zeroOfDType);
                var slice = Tensors.Slice(postPad, new[] { 0, 0, 0, 0 }, Tensors.ShapeOf(postPad), new[] { 0, 1, 2, 3 }, new[] { 1, 1, strides[0], strides[1] });

                return slice;
            }
        }

        return null;
    }
}
