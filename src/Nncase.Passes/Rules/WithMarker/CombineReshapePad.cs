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
using Nncase.IR.Shapes;
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
        MaybeMarker(
            IsReshape(
                "reshape",
                "reshapeCall",
                _ => true,
                HasMarker(IsPad("pad", "padCall", _ => true, HasMarker(IsWildcard("input"), "marker"), IsPaddings("pads"), IsTensorConst("value")) with { TypePattern = HasFixedShape() }, "padOutMarker"),
                IsWildcard("shape")) with
            { TypePattern = HasFixedShape() },
            "outMarker");

    private Expr? GetReplace(Reshape reshape, Call reshapeCall, Pad pad, Call padCall, Expr input, Expr shape, Paddings pads, Expr value, Marker marker, IMatchResult result)
    {
        // only support pattern like melgan
        var reshapeRank = reshapeCall.CheckedShape.Rank;
        var padRank = padCall.CheckedShape.Rank;
        if (reshapeRank >= padRank
            && Enumerable.SequenceEqual(reshapeCall.CheckedShape.ToValueArray()[(reshapeRank - padRank)..], padCall.CheckedShape.ToValueArray()))
        {
            var newPad = Pad(
                marker.With(target: Reshape(
                        marker.With(target: input),
                        Enumerable.Repeat(1L, reshapeRank - padRank).Concat(input.CheckedShape.ToValueArray()).ToArray())
                    .InheritMetaData(reshapeCall)),
                Enumerable.Repeat(Padding.Zero, (reshapeRank - padRank) * 2).Concat(pads).ToArray(),
                pad.PadMode,
                value).InheritMetaData(padCall);
            var outMarker = result.GetValueOrDefault("outMarker");
            if (outMarker != null)
            {
                return ((Marker)outMarker).With(target: newPad);
            }

            return newPad;
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
    public IPattern Pattern { get; } = MaybeMarker(
        IsPad(
        "pad",
        "padCall",
        x => true,
        HasMarker(IsTranspose(IsWildcard("input"), IsTensorConst("perm")), "marker"),
        IsPaddings("pads"),
        IsWildcard("padValue")),
        "outMarker");

    private Expr GetReplace(Pad pad, Call padCall, Expr input, int[] perm, Paddings pads, Expr padValue, Marker marker, IMatchResult result)
    {
        var inv_perm = perm.Select((p, i) => (p, i)).OrderBy(tp => tp.p).ToArray();
        var newPads = new List<Padding>();
        for (var i = 0; i < inv_perm.Length; i++)
        {
            newPads.Add(pads[inv_perm[i].i]);

            // newPads[i] = pads[perm[i]];
        }

        var p = Pad(input, newPads.ToArray(), pad.PadMode, padValue).InheritMetaData(padCall);
        var newTranspose = Transpose(marker.With(target: p), perm);
        var outMarker = result.GetValueOrDefault("outMarker");
        if (outMarker != null)
        {
            return ((Marker)outMarker).With(target: newTranspose);
        }

        return newTranspose;
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
    public IPattern Pattern { get; } = MaybeMarker(
        IsTranspose(
        "transpose",
        x => true,
        HasMarker(
            IsPad(
            "pad",
            "padCall",
            y => true,
            IsWildcard("input"),
            IsPaddings("pads"),
            IsTensorConst("padValue")),
            "marker"),
        IsTensorConst("perm")),
        "outMarker");

    private Expr GetReplace(Pad pad, Call padCall, Expr input, int[] perm, Paddings pads, Expr padValue, Marker marker, IMatchResult result)
    {
        var newPads = new List<Padding>();
        for (int i = 0; i < perm.Length; i++)
        {
            newPads.Add(pads[perm[i]]);
        }

        var newPad = Pad(
            marker.With(target: Transpose(input, perm)),
            newPads.ToArray(),
            pad.PadMode,
            padValue).InheritMetaData(padCall);
        var outMarker = result.GetValueOrDefault("outMarker");
        if (outMarker != null)
        {
            return ((Marker)outMarker).With(target: newPad);
        }

        return newPad;
    }
}
