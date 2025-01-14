// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.NN;
using Nncase.IR.Tensors;
using Nncase.PatternMatch;
using static Nncase.IR.F.NN;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.F.NN;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;
using static Nncase.Utilities.MetadataUtility;
using Shape = Nncase.IR.Shape;

namespace Nncase.Passes.Rules.Neutral;

// rules in this file are used for ShapeBucket

/// <summary>
/// Transform <see cref="IR.NN.Conv2D"/> to <see cref="IR.Math.Binary"/>.
/// </summary>
[RuleGenerator]
public sealed partial class FoldConv2DBiasWithMarker : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } = IsRangeOfMarker(
        "binarym",
        IsBinary(
            "binary",
            "binaryCall",
            p => p.BinaryOp is BinaryOp.Add,
            IsReshape(
                IsRangeOfMarker(
                    "convm",
                    IsConv2D(
                        "conv2d",
                        _ => true,
                        IsWildcard("input"),
                        IsWildcard("weights"),
                        IsTensorConst("bias") with { TypePattern = HasRank(1) },
                        IsWildcard("stride"),
                        IsWildcard("padding"),
                        IsWildcard("dilation"),
                        IsWildcard("groups")),
                    IsWildcard()),
                IsWildcard("shape")),
            IsRangeOfMarker("bm", IsTensorConst("b") with { TypePattern = HasRank(1) }, IsWildcard())),
        IsWildcard());

    private Expr? GetReplace(Conv2D conv2d, Call binaryCall, Expr input, Expr weights, Tensor bias, Tensor b, Expr shape, Expr stride, Expr padding, Expr dilation, Expr groups, Marker binarym)
    {
        var newBias = IR.F.Math.Add(bias, b).Evaluate().AsTensor();
        var newConv2d = Conv2D(
            input,
            weights,
            newBias,
            stride,
            padding,
            dilation,
            conv2d.PadMode,
            groups).InheritMetaData(binaryCall);
        var m = Reshape(binarym.With(target: newConv2d), shape).InheritMetaData(binaryCall);
        return m;
    }
}
