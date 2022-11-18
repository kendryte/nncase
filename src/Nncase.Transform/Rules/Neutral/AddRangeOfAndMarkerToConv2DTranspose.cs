
// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Nncase.IR;
using Nncase.IR.NN;
using Nncase.PatternMatch;
using Tensorflow;
using static Nncase.IR.F.NN;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.F.NN;
using static Nncase.PatternMatch.Utility;
using Shape = Nncase.IR.Shape;

namespace Nncase.Transform.Rules.Neutral;

/// <summary>
/// Insert RangeOf and RangeOfMarker
/// </summary>
[RuleGenerator]
public sealed partial class AddRangeOfAndMarkerToConv2DTranspose : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } =
        IsConv2DTranspose("conv2dTranspose", "call", _ => true,
            IsWildcard("input"),
            IsTensorConst("weights"),
            IsTensorConst("bias"),
            IsWildcard("outShape"),
            IsWildcard("stride"),
            IsWildcard("padding"),
            IsWildcard("outPadding"), 
            IsWildcard("dilation"),
            IsWildcard("groups"),
            IsWildcard("fusedClamp"));
    private Expr? GetReplace(Conv2DTranspose conv2dTranspose, Expr input, Expr weights, TensorConst bias, Expr outShape, Expr stride, Expr padding, 
        Expr outPadding, Expr dilation, Expr groups, Expr fusedClamp)
    {
        var output = Conv2DTranspose(IR.F.Math.RangeOfMarker(input, IR.F.Math.RangeOf(input)), IR.F.Math.RangeOfMarker(weights, IR.F.Math.RangeOf(weights)),
            bias, outShape, stride, padding, outPadding, dilation, PadMode.Constant, groups);
        return IR.F.Math.RangeOfMarker(output, IR.F.Math.RangeOf(output));
    }
}