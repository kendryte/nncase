// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Nncase.Evaluator;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.NN;
using Nncase.IR.Tensors;
using Nncase.PatternMatch;
using static Nncase.IR.F.Math;
using static Nncase.IR.F.NN;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.F.NN;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;
using static Nncase.Utilities.MetadataUtility;
using Tuple = System.Tuple;

namespace Nncase.Passes.Rules.WithMarker;

/// <summary>
/// reshape(pad(input), shape) => pad(reshape(input, shape)).
/// </summary>
[RuleGenerator]
public sealed partial class CombineReshapePad : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } =
        IsReshape(
                "reshape",
                "reshapeCall",
                _ => true,
                HasMarker(IsPad("pad", "padCall", _ => true, HasMarker(IsWildcard("input"), "marker"), IsTensorConst("pads"), IsTensorConst("value")) with { TypePattern = HasFixedShape() }, "padOutMarker"),
                IsWildcard("shape")) with
        { TypePattern = HasFixedShape() };

    private Expr? GetReplace(Reshape reshape, Call reshapeCall, Pad pad, Call padCall, Expr input, Expr shape, int[] pads, Expr value, Marker marker)
    {
        // only support pattern like melgan
        var reshapeRank = reshapeCall.CheckedShape.Rank;
        var padRank = padCall.CheckedShape.Rank;
        if (reshapeRank >= padRank
            && Enumerable.SequenceEqual(reshapeCall.CheckedShape.ToValueArray()[(reshapeRank - padRank)..], padCall.CheckedShape.ToValueArray()))
        {
            return Pad(
                marker.With(target: Reshape(input, Enumerable.Repeat(1, reshapeRank - padRank).Concat(input.CheckedShape.ToValueArray()).ToArray()).InheritMetaData(reshapeCall)),
                Tensor.From(Enumerable.Repeat(0, (reshapeRank - padRank) * 2).Concat(pads).ToArray(), new[] { reshapeRank, 2 }),
                pad.PadMode,
                value).InheritMetaData(padCall);
        }

        return null;
    }
}

/// <summary>
/// Combine Transpose with Pad
/// pad(transpose(x,p), pp) => transpose(pad(x, invtranspose(pp, p)), p).
/// </summary>
[RuleGenerator]
public sealed partial class CombineTransposePad : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } = IsPad(
        "pad",
        "padCall",
        x => true,
        HasMarker(IsTranspose(IsWildcard("input"), IsTensorConst("perm")), "marker"),
        IsTensorConst("pads"),
        IsWildcard("padValue"));

    private Expr GetReplace(Pad pad, Call padCall, Expr input, int[] perm, Expr pads, Expr padValue, Marker marker)
    {
        var inv_perm = perm.Select((p, i) => (p, i)).OrderBy(tp => tp.p).ToArray();
        var newPads = new List<Expr>();
        for (var i = 0; i < inv_perm.Length; i++)
        {
            newPads.Add(Stack(new IR.Tuple(pads[inv_perm[i].i, 0], pads[inv_perm[i].i, 1]), 0));

            // newPads[i] = pads[perm[i]];
        }

        var p = Pad(input, Stack(new IR.Tuple(newPads.ToArray()), 0).Evaluate().AsTensor(), pad.PadMode, padValue).InheritMetaData(padCall);
        return Transpose(marker.With(target: p), perm);
    }
}

/// <summary>
/// Combine Pad with Transpose
/// transpose(pad(x, pp),p) => pad(transpose(x),new_pp).
/// </summary>
[RuleGenerator]
public sealed partial class CombinePadTranspose : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } = IsTranspose(
        "transpose",
        x => true,
        HasMarker(
            IsPad(
            "pad",
            "padCall",
            y => true,
            IsWildcard("input"),
            IsTensorConst("pads"),
            IsTensorConst("padValue")),
            "marker"),
        IsTensorConst("perm"));

    private Expr GetReplace(Pad pad, Call padCall, Expr input, int[] perm, Expr pads, Expr padValue, Marker marker)
    {
        var newPads = new List<int>();
        for (int i = 0; i < perm.Length; i++)
        {
            newPads.Add(((TensorConst)pads).Value.ToArray<int>()[perm[i] * 2]);
            newPads.Add(((TensorConst)pads).Value.ToArray<int>()[(perm[i] * 2) + 1]);
        }

        return Pad(marker.With(target: Transpose(input, perm)), Tensor.From<int>(newPads.ToArray(), pads.CheckedShape), pad.PadMode, padValue).InheritMetaData(padCall);
    }
}
