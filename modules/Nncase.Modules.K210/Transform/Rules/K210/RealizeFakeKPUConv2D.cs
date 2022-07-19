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
using static  Nncase.PatternMatch.F.K210;
using Nncase.PatternMatch;
using Nncase.Utilities;
using Tensorflow.Keras;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.F.NN;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Transform.Rules.K210;

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
            IsRangeOfMarker(IsWildcard("input"), IsConst("input_range")),
            IsTensorConst("weights"));

    private Expr? GetReplace(Expr fake_conv2d_call, Expr input, Tensor<float> input_range, Expr weights)
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
            var kpuConv = IR.F.K210.KPUConv2D(isDepthwise, filterType, KPUPoolType.Bypass, new KPUActivationParameters(), input);
            return kpuConv;
        }
        return null;
    }
}