// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.IR.K210;
using Nncase.PatternMatch;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.NN;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Transform.Rules.K210;

/// <summary>
/// Lower <see cref="IR.NN.Conv2D"/> to <see cref="IR.K210.FakeKPUConv2D"/>.
/// </summary>
[RuleGenerator]
public sealed class LowerConv2D : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } = IsConv2D(
        null,
        "conv2d",
        PadMode.Constant,
        IsWildcard("input") with { TypePattern = HasFixedShape() },
        IsWildcard("weights") with { TypePattern = HasFixedShape() },
        IsTensorConst("bias"),
        IsTensorConst("strides"),
        IsTensorConst("paddings"),
        new[] { 1, 1 },
        IsTensorConst("groups"),
        IsTensorConst("fusedClamp")) with
    {
        TypePattern = HasFixedShape(),
    };

    private Expr? GetReplace(Expr conv2d, Expr input, Expr weights, Tensor<float> bias, int[] strides, int[] paddings, int groups, float[] fusedClamp)
    {
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

        }

        return null;
    }
}
