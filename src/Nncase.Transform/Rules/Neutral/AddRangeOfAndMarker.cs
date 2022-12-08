
// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Nncase.IR;
using Nncase.IR.Imaging;
using Nncase.IR.Math;
using Nncase.IR.NN;
using Nncase.IR.Tensors;
using Nncase.PatternMatch;
using Tensorflow;
using static Nncase.IR.F.Imaging;
using static Nncase.IR.F.Math;
using static Nncase.IR.F.NN;
using static Nncase.IR.F.RNN;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.F.NN;
using static Nncase.PatternMatch.F.Imaging;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;
using Binary = Nncase.IR.Math.Binary;
using Shape = Nncase.IR.Shape;

namespace Nncase.Transform.Rules.Neutral;

/// <summary>
/// Insert RangeOf and RangeOfMarker
/// </summary>

[RuleGenerator]
public sealed partial class AddRangeOfAndMarkerToBatchToSpace : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } =
        IsBatchToSpace("batchtospace", "call", _ => true,
            IsWildcard("input"),
            IsWildcard("blockshape"),
            IsWildcard("crops"));
    private Expr? GetReplace(BatchToSpace batchtospace, Call call, Expr input, Expr blockshape, TensorConst crops, RunPassOptions options)
    {
        var output = BatchToSpace(IR.F.Math.RangeOfMarker(input, IR.F.Math.RangeOf(input)), blockshape, crops);
        options.MatchOptions.SuppressPattern(output, Pattern); //only invoke once
        return IR.F.Math.RangeOfMarker(output, IR.F.Math.RangeOf(output));
    }
}

[RuleGenerator]
public sealed partial class AddRangeOfAndMarkerToBinary : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } =
        IsBinary("binary", "call",
            _ => true,
            IsWildcard("lhs"),
            IsWildcard("rhs"));
    private Expr? GetReplace(Binary binary, Call call, Expr lhs, Expr rhs, RunPassOptions options)
    {
        var output = Nncase.IR.F.Math.Binary(binary.BinaryOp, IR.F.Math.RangeOfMarker(lhs, IR.F.Math.RangeOf(lhs)), IR.F.Math.RangeOfMarker(rhs, IR.F.Math.RangeOf(rhs)));
        options.MatchOptions.SuppressPattern(output, Pattern); //only invoke once
        return IR.F.Math.RangeOfMarker(output, IR.F.Math.RangeOf(output));
    }
}

[RuleGenerator]
public sealed partial class AddRangeOfAndMarkerToBroadcast : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } =
        IsBroadcast("broadcast", _ => true,
            IsWildcard("input"),
            IsTensorConst("shape"));
    private Expr? GetReplace(Broadcast broadcast, Expr input, Expr shape, RunPassOptions options)
    {
        var output = Broadcast(IR.F.Math.RangeOfMarker(input, IR.F.Math.RangeOf(input)), shape);
        options.MatchOptions.SuppressPattern(output, Pattern); //only invoke once
        return IR.F.Math.RangeOfMarker(output, IR.F.Math.RangeOf(output));
    }
}

[RuleGenerator]
public sealed partial class AddRangeOfAndMarkerToCelu : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } =
        IsCelu("celu", "call", _ => true,
            IsWildcard("input"),
            IsWildcard("alpha"));

    private Expr? GetReplace(Celu celu, Call call, Expr input, Expr alpha, RunPassOptions options)
    {
        var output = Celu(IR.F.Math.RangeOfMarker(input, IR.F.Math.RangeOf(input)), alpha);
        options.MatchOptions.SuppressPattern(output, Pattern); //only invoke once
        return IR.F.Math.RangeOfMarker(output, IR.F.Math.RangeOf(output));
    }
}

[RuleGenerator]
public sealed partial class AddRangeOfAndMarkerToCompare : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } =
        IsCompare("compare", "call", _ => true,
            IsWildcard("lhs"),
            IsWildcard("rhs"));

    private Expr? GetReplace(Compare compare, Call call, Expr lhs, Expr rhs, RunPassOptions options)
    {
        var output = Compare(compare.CompareOp, IR.F.Math.RangeOfMarker(lhs, IR.F.Math.RangeOf(lhs)), IR.F.Math.RangeOfMarker(rhs, IR.F.Math.RangeOf(rhs)));
        options.MatchOptions.SuppressPattern(output, Pattern); //only invoke once
        return IR.F.Math.RangeOfMarker(output, IR.F.Math.RangeOf(output));
    }
}

[RuleGenerator]
public sealed partial class AddRangeOfAndMarkerToConv2D : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } =
        IsConv2D("conv", "call", _ => true,
            IsWildcard("input"),
            IsTensorConst("weights"),
            IsTensorConst("bias"),
            IsTensorConst("stride"),
            IsWildcard("padding"),
            IsTensorConst("dilation"),
            IsTensorConst("groups"),
            IsWildcard("fusedClamp"));
    private Expr? GetReplace(Conv2D conv, Call call, Expr input, Expr weights, TensorConst bias, Expr stride, Expr padding,
        Expr dilation, Expr groups, Expr fusedClamp)
    {
        var output = Conv2D(IR.F.Math.RangeOfMarker(input, IR.F.Math.RangeOf(input)), IR.F.Math.RangeOfMarker(weights, IR.F.Math.RangeOf(weights)),
            bias, stride, padding, dilation, PadMode.Constant, groups, fusedClamp);
        return IR.F.Math.RangeOfMarker(output, IR.F.Math.RangeOf(output));
    }
}

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
            IsTensorConst("stride"),
            IsWildcard("padding"),
            IsWildcard("outPadding"),
            IsTensorConst("dilation"),
            IsTensorConst("groups"),
            IsWildcard("fusedClamp"));
    private Expr? GetReplace(Conv2DTranspose conv2dTranspose, Expr input, Expr weights, TensorConst bias, Expr outShape, Expr stride, Expr padding,
        Expr outPadding, Expr dilation, Expr groups, Expr fusedClamp, RunPassOptions options)
    {
        var output = Conv2DTranspose(IR.F.Math.RangeOfMarker(input, IR.F.Math.RangeOf(input)), IR.F.Math.RangeOfMarker(weights, IR.F.Math.RangeOf(weights)),
            bias, outShape, stride, padding, outPadding, dilation, PadMode.Constant, groups);
        options.MatchOptions.SuppressPattern(output, Pattern); //only invoke once
        return IR.F.Math.RangeOfMarker(output, IR.F.Math.RangeOf(output));
    }
}

[RuleGenerator]
public sealed partial class AddRangeOfAndMarkerToElu : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } =
        IsElu("elu", "call", _ => true,
            IsWildcard("input"),
            IsWildcard("alpha"));

    private Expr? GetReplace(Elu elu, Call call, Expr input, Expr alpha, RunPassOptions options)
    {
        var output = Elu(IR.F.Math.RangeOfMarker(input, IR.F.Math.RangeOf(input)), alpha);
        options.MatchOptions.SuppressPattern(output, Pattern); //only invoke once
        return IR.F.Math.RangeOfMarker(output, IR.F.Math.RangeOf(output));
    }
}

[RuleGenerator]
public sealed partial class AddRangeOfAndMarkerToHardMax : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } =
        IsHardmax("hardmax", "call", _ => true,
            IsWildcard("input"),
            IsWildcard("axis"));
    private Expr? GetReplace(Hardmax hardmax, Call call, Expr input, Expr axis, RunPassOptions options)
    {
        var output = Hardmax(IR.F.Math.RangeOfMarker(input, IR.F.Math.RangeOf(input)), axis);
        options.MatchOptions.SuppressPattern(output, Pattern); //only invoke once
        return IR.F.Math.RangeOfMarker(output, IR.F.Math.RangeOf(output));
    }
}

[RuleGenerator]
public sealed partial class AddRangeOfAndMarkerToHardSigmoid : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } =
        IsHardSigmoid("hardSigmoid", "call", _ => true,
            IsWildcard("input"),
            IsWildcard("alpha"),
            IsWildcard("beta"));

    private Expr? GetReplace(HardSigmoid hardSigmoid, Call call, Expr input, Expr alpha, Expr beta, RunPassOptions options)
    {
        var output = HardSigmoid(IR.F.Math.RangeOfMarker(input, IR.F.Math.RangeOf(input)), alpha, beta);
        options.MatchOptions.SuppressPattern(output, Pattern); //only invoke once
        return IR.F.Math.RangeOfMarker(output, IR.F.Math.RangeOf(output));
    }
}

[RuleGenerator]
public sealed partial class AddRangeOfAndMarkerToHardSwish : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } =
        IsHardSwish("hardSwish", "call", _ => true,
            IsWildcard("input"));

    private Expr? GetReplace(HardSwish hardSwish, Call call, Expr input, RunPassOptions options)
    {
        var output = HardSwish(IR.F.Math.RangeOfMarker(input, IR.F.Math.RangeOf(input)));
        options.MatchOptions.SuppressPattern(output, Pattern); //only invoke once
        return IR.F.Math.RangeOfMarker(output, IR.F.Math.RangeOf(output));
    }
}

[RuleGenerator]
public sealed partial class AddRangeOfAndMarkerToLSTM : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } =
        IsLSTM("lstm", "call", _ => true,
            IsWildcard("x"),
            IsTensorConst("w"),
            IsTensorConst("r"),
            IsTensorConst("b"),
            IsTensorConst("sequencelens"),
            IsTensorConst("initialh"),
            IsTensorConst("initialc"),
            IsTensorConst("p"),
            IsTensorConst("actalpha"),
            IsTensorConst("actbeta"),
            IsTensorConst("clip"),
            IsTensorConst("hiddensize"),
            IsTensorConst("inputforget"),
            IsTensorConst("outputsize"));

    private Expr? GetReplace(IR.Tensors.LSTM lstm, Call call, Expr x, TensorConst w, TensorConst r, TensorConst b, TensorConst sequencelens,
                            TensorConst initialh, TensorConst initialc, TensorConst p, TensorConst actalpha, TensorConst actbeta,
                            TensorConst clip, TensorConst hiddensize, TensorConst inputforget, TensorConst outputsize, RunPassOptions options)
    {
        var output = LSTM(lstm.Direction, lstm.Layout, lstm.Activations, IR.F.Math.RangeOfMarker(x, IR.F.Math.RangeOf(x)), w, r, b, sequencelens, initialh, initialc,
                         p, actalpha, actbeta, clip, hiddensize, inputforget, outputsize);
        options.MatchOptions.SuppressPattern(output, Pattern); //only invoke once
        return IR.F.Math.RangeOfMarker(output, IR.F.Math.RangeOf(output));
    }
}

[RuleGenerator]
public sealed partial class AddRangeOfAndMarkerToMatMul : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } =
        IsMatMul("matmul", "call", _ => true,
            IsWildcard("lhs"),
            IsWildcard("rhs"));
    private Expr? GetReplace(MatMul matmul, Call call, Expr lhs, Expr rhs, RunPassOptions options)
    {
        var output = Nncase.IR.F.Math.MatMul(IR.F.Math.RangeOfMarker(lhs, IR.F.Math.RangeOf(lhs)), IR.F.Math.RangeOfMarker(rhs, IR.F.Math.RangeOf(rhs)));
        options.MatchOptions.SuppressPattern(output, Pattern); //only invoke once
        return IR.F.Math.RangeOfMarker(output, IR.F.Math.RangeOf(output));
    }
}

[RuleGenerator]
public sealed partial class AddRangeOfAndMarkerToPad : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } =
        IsPad("pad", "call", _ => true,
            IsWildcard("input"),
            IsWildcard("pads"),
            IsWildcard("value"));

    private Expr? GetReplace(Pad pad, Call call, Expr input, Expr pads, Expr value, RunPassOptions options)
    {
        var output = Pad(IR.F.Math.RangeOfMarker(input, IR.F.Math.RangeOf(input)), pads, pad.PadMode, value);
        options.MatchOptions.SuppressPattern(output, Pattern); //only invoke once
        return IR.F.Math.RangeOfMarker(output, IR.F.Math.RangeOf(output));
    }
}
[RuleGenerator]
public sealed partial class AddRangeOfAndMarkerToPRelu : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } =
        IsPRelu("prelu", "call", _ => true,
            IsWildcard("input"),
            IsWildcard("slope"));

    private Expr? GetReplace(PRelu prelu, Call call, Expr input, Expr slope, RunPassOptions options)
    {
        var output = PRelu(IR.F.Math.RangeOfMarker(input, IR.F.Math.RangeOf(input)), slope);
        options.MatchOptions.SuppressPattern(output, Pattern); //only invoke once
        return IR.F.Math.RangeOfMarker(output, IR.F.Math.RangeOf(output));
    }
}

[RuleGenerator]
public sealed partial class AddRangeOfAndMarkerToRedece : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } =
        IsReduce("reduce", "call", _ => true,
            IsWildcard("input"),
            IsTensorConst("axis"),
            IsWildcard("initValue"),
            IsTensorConst("keepDims"));

    private Expr? GetReplace(Reduce reduce, Expr input, TensorConst axis, Expr initValue, TensorConst keepDims, RunPassOptions options)
    {
        var output = Reduce(reduce.ReduceOp, IR.F.Math.RangeOfMarker(input, IR.F.Math.RangeOf(input)), axis, initValue, keepDims);
        options.MatchOptions.SuppressPattern(output, Pattern); //only invoke once
        return IR.F.Math.RangeOfMarker(output, IR.F.Math.RangeOf(output));
    }
}

[RuleGenerator]
public sealed partial class AddRangeOfAndMarkerToRedeceWindow2D : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } =
        IsReduceWindow2D("reduceWindow2D", "call", _ => true,
            IsWildcard("input"),
            IsWildcard("initvalue"),
            IsWildcard("filter"),
            IsWildcard("stride"),
            IsWildcard("padding"),
            IsWildcard("dilation"),
            IsWildcard("ceilmode"),
            IsWildcard("countincludepad"));

    private Expr? GetReplace(ReduceWindow2D reduceWindow2D, Expr input, Expr initvalue, Expr filter, Expr stride, Expr padding,
        Expr dilation, Expr ceilmode, Expr countincludepad, RunPassOptions options)
    {
        var output = ReduceWindow2D(reduceWindow2D.ReduceOp, IR.F.Math.RangeOfMarker(input, IR.F.Math.RangeOf(input)), initvalue, filter, stride, padding, dilation, ceilmode, countincludepad);
        options.MatchOptions.SuppressPattern(output, Pattern); //only invoke once
        return IR.F.Math.RangeOfMarker(output, IR.F.Math.RangeOf(output));
    }
}

[RuleGenerator]
public sealed partial class AddRangeOfAndMarkerToResizeImage : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } =
        IsResizeImage("resize", "call", _ => true,
            IsWildcard("input"),
            IsWildcard("roi"),
            IsTensorConst("newSize"),
            IsWildcard("cubiccoeffa"),
            IsWildcard("excludeOutside"),
            IsWildcard("extrapolationValue"));
    private Expr? GetReplace(ResizeImage resize, Call call, Expr input, Expr roi, TensorConst newSize, Expr cubiccoeffa, Expr excludeOutside, Expr extrapolationValue, RunPassOptions options)
    {
        var output = ResizeImage(resize.ResizeMode, resize.TransformationMode, resize.NearestMode, IR.F.Math.RangeOfMarker(input, IR.F.Math.RangeOf(input)), roi, newSize, cubiccoeffa, excludeOutside, extrapolationValue, resize.IsTFResize);
        options.MatchOptions.SuppressPattern(output, Pattern); //only invoke once
        return IR.F.Math.RangeOfMarker(output, IR.F.Math.RangeOf(output));
    }
}

[RuleGenerator]
public sealed partial class AddRangeOfAndMarkerToSelu : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } =
        IsSelu("selu", "call", _ => true,
            IsWildcard("input"),
            IsWildcard("alpha"),
            IsWildcard("gamma"));

    private Expr? GetReplace(Selu selu, Call call, Expr input, Expr alpha, Expr gamma, RunPassOptions options)
    {
        var output = Selu(IR.F.Math.RangeOfMarker(input, IR.F.Math.RangeOf(input)), alpha, gamma);
        options.MatchOptions.SuppressPattern(output, Pattern); //only invoke once
        return IR.F.Math.RangeOfMarker(output, IR.F.Math.RangeOf(output));
    }
}

[RuleGenerator]
public sealed partial class AddRangeOfAndMarkerToSigmoid : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } =
        IsSigmoid("sigmoid", "call", _ => true,
            IsWildcard("input"));
    private Expr? GetReplace(Sigmoid sigmoid, Call call, Expr input, RunPassOptions options)
    {
        var output = Sigmoid(IR.F.Math.RangeOfMarker(input, IR.F.Math.RangeOf(input)));
        options.MatchOptions.SuppressPattern(output, Pattern); //only invoke once
        return IR.F.Math.RangeOfMarker(output, IR.F.Math.RangeOf(output));
    }
}

[RuleGenerator]
public sealed partial class AddRangeOfAndMarkerToSpaceToBatch : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } =
        IsSpaceToBatch("spacetobatch", "call", _ => true,
            IsWildcard("input"),
            IsWildcard("blockshape"),
            IsWildcard("paddings"));
    private Expr? GetReplace(SpaceToBatch spacetobatch, Call call, Expr input, Expr blockshape, TensorConst paddings, RunPassOptions options)
    {
        var output = SpaceToBatch(IR.F.Math.RangeOfMarker(input, IR.F.Math.RangeOf(input)), blockshape, paddings);
        options.MatchOptions.SuppressPattern(output, Pattern); //only invoke once
        return IR.F.Math.RangeOfMarker(output, IR.F.Math.RangeOf(output));
    }
}

[RuleGenerator]
public sealed partial class AddRangeOfAndMarkerToTranspose : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } =
        IsTranspose("transpose", "call", _ => true,
            IsWildcard("input"),
            IsTensorConst("perm"));

    private Expr? GetReplace(Transpose transpose, Expr input, TensorConst perm, RunPassOptions options)
    {
        var output = Transpose(IR.F.Math.RangeOfMarker(input, IR.F.Math.RangeOf(input)), perm);
        options.MatchOptions.SuppressPattern(output, Pattern); //only invoke once
        return IR.F.Math.RangeOfMarker(output, IR.F.Math.RangeOf(output));
    }
}

[RuleGenerator]
public sealed partial class AddRangeOfAndMarkerToUnary : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } =
        IsUnary("unary", "call",
            _ => true,
            IsWildcard("input"));
    private Expr? GetReplace(Unary unary, Call call, Expr input, RunPassOptions options)
    {
        var output = Nncase.IR.F.Math.Unary(unary.UnaryOp, IR.F.Math.RangeOfMarker(input, IR.F.Math.RangeOf(input)));
        options.MatchOptions.SuppressPattern(output, Pattern); //only invoke once
        return IR.F.Math.RangeOfMarker(output, IR.F.Math.RangeOf(output));
    }
}
