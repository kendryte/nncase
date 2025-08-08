// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using DryIoc.ImTools;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.PatternMatch;
using Nncase.Utilities;

using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Imaging;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.F.NN;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.NTT;

[RuleGenerator]
public sealed class PackResizeImagePropagation : RewriteRule<Pattern>
{
    public override Pattern Pattern { get; } =
        PatternMatch.F.Tensors.IsPack(
            "pack",
            "caller",
            _ => true,
            IsResizeImage(
                "resize",
                "callee",
                op => op.TransformationMode == ImageResizeTransformationMode.Asymmetric && op.IsTFResize == false,
                IsWildcard("input") with { TypePattern = !IsVector() },
                IsWildcard("roi"),
                IsTensorConst("newSize"),
                IsTensorConst("cubicCoeffA"),
                IsTensorConst("excludeOutside"),
                IsTensorConst("extrapolationValue")));

    public override Expr? GetReplace(IMatchResult result, RunPassContext context)
    {
        var pack = (IR.Tensors.Pack)result["pack"];
        if (pack.Lanes.Count > 1)
        {
            var op = (IR.Imaging.ResizeImage)result["resize"];
            var input = (Expr)result["input"];
            var newSize = ((TensorConst)result["newSize"]).Value.ToArray<int>();
            var ret = PackResizeImage.AddCandidate(op, input, newSize, pack.Axes.ToArray(), pack.Lanes.ToArray()).FirstOrDefault();
            if (ret is not null)
            {
                return IR.F.Tensors.Pack(ret, pack.Lanes.ToArray(), pack.Axes.ToArray());
            }
        }

        return null;
    }
}

[RuleGenerator]
public sealed class ResizeUnpackImagePropagation : RewriteRule<Pattern>
{
    public override Pattern Pattern { get; } =
        IsResizeImage(
            "resize",
            "caller",
            op => op.TransformationMode == ImageResizeTransformationMode.Asymmetric && op.IsTFResize == false,
            PatternMatch.F.Tensors.IsUnpack(
                "unpack",
                "callee",
                _ => true,
                IsWildcard("input")),
            IsWildcard("roi"),
            IsTensorConst("newSize"),
            IsTensorConst("cubicCoeffA"),
            IsTensorConst("excludeOutside"),
            IsTensorConst("extrapolationValue"));

    public override Expr? GetReplace(IMatchResult result, RunPassContext context)
    {
        var unpack = (IR.Tensors.Unpack)result["unpack"];
        if (unpack.Lanes.Count > 1)
        {
            var op = (IR.Imaging.ResizeImage)result["resize"];
            var callee = (Expr)result["callee"];
            var newSize = ((TensorConst)result["newSize"]).Value.ToArray<int>();
            var ret = PackResizeImage.AddCandidate(op, callee, newSize, unpack.Axes.ToArray(), unpack.Lanes.ToArray()).FirstOrDefault();
            if (ret is not null)
            {
                return ret;
            }
        }

        return null;
    }
}

[RuleGenerator]
public sealed class PackReducePropagation : RewriteRule<Pattern>
{
    public override Pattern Pattern { get; } =
        PatternMatch.F.Tensors.IsPack(
            "pack",
            "caller",
            _ => true,
            IsReduce(
                "reduce",
                "callee",
                r => r.ReduceOp is ReduceOp.Mean or ReduceOp.Sum,
                IsWildcard("input", e => e is not Call { Target: IR.Tensors.Unpack }) with { TypePattern = IsFloat() & !IsVector() },
                IsTensorConst("axes") with { TypePattern = IsIntegral() },
                IsTensorConst("initValue") with { TypePattern = IsFloat() },
                IsTensorConst("keepDims") with { TypePattern = IsBool() }));

    public override Expr? GetReplace(IMatchResult result, RunPassContext context)
    {
        var pack = (IR.Tensors.Pack)result["pack"];
        if (pack.Lanes.Count > 1)
        {
            var op = (IR.Math.Reduce)result["reduce"];
            var input = (Expr)result["input"];
            var axes = ((TensorConst)result["axes"]).Value.ToArray<int>();
            if (axes.Length > 1)
            {
                return null;
            }

            var initValue = ((TensorConst)result["initValue"]).Value.ToScalar<float>();
            var keepDims = ((TensorConst)result["keepDims"]).Value.ToScalar<bool>();
            var remainAxes = Enumerable.Range(0, input.CheckedShape.Rank).Where(i => !axes.Contains(i)).ToArray();
            var packAxes = keepDims ?
                pack.Axes.ToArray() :
                pack.Axes.Select(a => remainAxes[a]).ToArray();

            var ret = PackReduce.AddCandidate(op, input, axes, initValue, keepDims, packAxes, pack.Lanes.ToArray()).FirstOrDefault();
            if (ret is not null)
            {
                return IR.F.Tensors.Pack(ret, pack.Lanes.ToArray(), pack.Axes.ToArray());
            }
        }

        return null;
    }
}

[RuleGenerator]
public sealed class ReduceUnpackPropagation : RewriteRule<Pattern>
{
    public override Pattern Pattern { get; } =
        IsReduce(
            "reduce",
            "caller",
            r => r.ReduceOp is ReduceOp.Mean or ReduceOp.Sum,
            PatternMatch.F.Tensors.IsUnpack(
                "unpack",
                "callee",
                _ => true,
                IsWildcard("input")),
            IsTensorConst("axes") with { TypePattern = IsIntegral() },
            IsTensorConst("initValue") with { TypePattern = IsFloat() },
            IsTensorConst("keepDims") with { TypePattern = IsBool() });

    public override Expr? GetReplace(IMatchResult result, RunPassContext context)
    {
        var unpack = (IR.Tensors.Unpack)result["unpack"];
        if (unpack.Lanes.Count > 1)
        {
            var op = (IR.Math.Reduce)result["reduce"];
            var callee = (Expr)result["callee"];
            var axes = ((TensorConst)result["axes"]).Value.ToArray<int>();
            if (axes.Length > 1)
            {
                return null;
            }

            var initValue = ((TensorConst)result["initValue"]).Value.ToScalar<float>();
            var keepDims = ((TensorConst)result["keepDims"]).Value.ToScalar<bool>();
            var ret = PackReduce.AddCandidate(op, callee, axes, initValue, keepDims, unpack.Axes.ToArray(), unpack.Lanes.ToArray()).FirstOrDefault();
            if (ret is not null)
            {
                return ret;
            }
        }

        return null;
    }
}

[RuleGenerator]
public sealed class PackUnsqueezePropagation : RewriteRule<Pattern>
{
    public override Pattern Pattern { get; } =
        PatternMatch.F.Tensors.IsPack(
            "pack",
            "caller",
            _ => true,
            IsUnsqueeze(
                "unsq",
                "callee",
                IsWildcard("input", e => e is not Call { Target: IR.Tensors.Unpack }) with { TypePattern = IsFloat() & !IsVector() },
                IsTensorConst("axes") with { TypePattern = IsIntegral() }));

    public override Expr? GetReplace(IMatchResult result, RunPassContext context)
    {
        var pack = (IR.Tensors.Pack)result["pack"];
        if (pack.Axes.Count > 1)
        {
            var input = (Expr)result["input"];
            var axes = ((TensorConst)result["axes"]).Value.ToArray<int>();
            var packAxes = pack.Axes.Select(pa => pa - axes.Where(a => a < pa).Count()).ToArray();

            var ret = PackUnsqueeze.AddCandidate(input, axes, packAxes, pack.Lanes.ToArray()).FirstOrDefault();
            if (ret is not null)
            {
                return IR.F.Tensors.Pack(ret, pack.Lanes.ToArray(), pack.Axes.ToArray());
            }
        }

        return null;
    }
}

[RuleGenerator]
public sealed class UnsqueezeUnpackPropagation : RewriteRule<Pattern>
{
    public override Pattern Pattern { get; } =
        IsUnsqueeze(
            "unsq",
            "caller",
            PatternMatch.F.Tensors.IsUnpack(
                "unpack",
                "callee",
                _ => true,
                IsWildcard("input")),
            IsTensorConst("axes") with { TypePattern = IsIntegral() });

    public override Expr? GetReplace(IMatchResult result, RunPassContext context)
    {
        var unpack = (IR.Tensors.Unpack)result["unpack"];
        if (unpack.Axes.Count > 1)
        {
            var calee = (Expr)result["callee"];
            var axes = ((TensorConst)result["axes"]).Value.ToArray<int>();

            var ret = PackUnsqueeze.AddCandidate(calee, axes, unpack.Axes.ToArray(), unpack.Lanes.ToArray()).FirstOrDefault();
            if (ret is not null)
            {
                return ret;
            }
        }

        return null;
    }
}

[RuleGenerator]
public sealed class CastUnpackPropagation : RewriteRule<Pattern>
{
    public override Pattern Pattern { get; } =
    IsCast(
        "cast",
        "caller",
        _ => true,
        PatternMatch.F.Tensors.IsUnpack(
            "unpack",
            "callee",
            _ => true,
            IsWildcard("input")));

    public override Expr? GetReplace(IMatchResult result, RunPassContext context)
    {
        var unpack = (IR.Tensors.Unpack)result["unpack"];
        if (unpack.Axes.Count > 1)
        {
            var caller = (Call)result["caller"];
            var callee = (Expr)result["callee"];

            var ret = PackCast.AddCandidate(caller, callee, unpack.Axes.ToArray(), unpack.Lanes.ToArray()).FirstOrDefault();
            if (ret is not null)
            {
                return ret;
            }
        }

        return null;
    }
}

[RuleGenerator]
public sealed class PackComparePropagation : RewriteRule<Pattern>
{
    public PackComparePropagation(MaskVectorStyle maskVectorStyle)
    {
        MaskVectorStyle = maskVectorStyle;
    }

    public MaskVectorStyle MaskVectorStyle { get; }

    public override Pattern Pattern { get; } =
        PatternMatch.F.Tensors.IsPack(
            "pack",
            "caller",
            _ => true,
            IsCompare(
                "compare",
                "callee",
                _ => true,
                IsWildcard("lhs", e => e is not Call { Target: IR.Tensors.Unpack }) with { TypePattern = !IsVector() },
                IsWildcard("rhs", e => e is not Call { Target: IR.Tensors.Unpack }) with { TypePattern = !IsVector() }));

    public override Expr? GetReplace(IMatchResult result, RunPassContext context)
    {
        var pack = (IR.Tensors.Pack)result["pack"];
        if (pack.Lanes.Count > 1)
        {
            var op = (IR.Math.Compare)result["compare"];
            var lhs = (Expr)result["lhs"];
            var rhs = (Expr)result["rhs"];
            var lhsShape = lhs.CheckedShape;
            var rhsShape = rhs.CheckedShape;
            var candidate = (Expr)result[Pattern];
            var lhsExt = candidate.CheckedShape.Rank - lhsShape.Rank;
            var rhsExt = candidate.CheckedShape.Rank - rhsShape.Rank;
            var lhsPackedAxes = pack.Axes.Where(a => a - lhsExt >= 0 && lhsShape[a - lhsExt] is { IsFixed: true } fa && fa != 1).ToArray();
            var rhsPackedAxes = pack.Axes.Where(a => a - rhsExt >= 0 && rhsShape[a - rhsExt] is { IsFixed: true } fa && fa != 1).ToArray();
            var lhsLanes = lhsPackedAxes.Select(a => pack.Axes.IndexOf(a)).Select(i => pack.Lanes[i]).ToArray();
            var rhsLanes = rhsPackedAxes.Select(a => pack.Axes.IndexOf(a)).Select(i => pack.Lanes[i]).ToArray();

            var ret = PackCompare.AddCandidate(op, lhs, rhs, candidate, lhsPackedAxes.Select(a => a - lhsExt).ToArray(), rhsPackedAxes.Select(a => a - rhsExt).ToArray(), lhsLanes, rhsLanes, MaskVectorStyle).FirstOrDefault();
            if (ret is not null)
            {
                return IR.F.Tensors.Pack(ret, pack.Lanes.ToArray(), pack.Axes.ToArray());
            }
        }

        return null;
    }
}

[RuleGenerator]
public sealed class CompareUnpackPropagation : RewriteRule<Pattern>
{
    public CompareUnpackPropagation(MaskVectorStyle maskVectorStyle)
    {
        MaskVectorStyle = maskVectorStyle;
    }

    public MaskVectorStyle MaskVectorStyle { get; }

    public override Pattern Pattern { get; } =
    IsCompare(
            "compare",
            "caller",
            _ => true,
            IsAlt(
                PatternMatch.F.Tensors.IsUnpack("lhsUnpack", "lhs", _ => true, IsWildcard("lhsIn")),
                IsWildcard("lhs")),
            IsAlt(
                PatternMatch.F.Tensors.IsUnpack("rhsUnpack", "rhs", _ => true, IsWildcard("rhsIn")),
                IsWildcard("rhs")));

    public override Expr? GetReplace(IMatchResult result, RunPassContext context)
    {
        var lhsUnpack = result.GetValueOrDefault("lhsUnpack") as IR.Tensors.Unpack;
        var rhsUnpack = result.GetValueOrDefault("rhsUnpack") as IR.Tensors.Unpack;
        if ((lhsUnpack != null && lhsUnpack.Lanes.Count > 1) || (rhsUnpack != null && rhsUnpack.Lanes.Count > 1))
        {
            var op = (IR.Math.Compare)result["compare"];
            var lhs = (Expr)result["lhs"];
            var rhs = (Expr)result["rhs"];
            var candidate = (Expr)result[Pattern];
            var lhsPackedAxes = (lhsUnpack != null && lhsUnpack.Lanes.Count > 1) ? lhsUnpack.Axes : Array.Empty<int>();
            var rhsPackedAxes = (rhsUnpack != null && rhsUnpack.Lanes.Count > 1) ? rhsUnpack.Axes : Array.Empty<int>();
            var lhsLanes = lhsPackedAxes.Count > 0 ? lhsUnpack!.Lanes : Array.Empty<int>();
            var rhsLanes = rhsPackedAxes.Count > 0 ? rhsUnpack!.Lanes : Array.Empty<int>();

            var ret = PackCompare.AddCandidate(op, lhs, rhs, candidate, lhsPackedAxes.ToArray(), rhsPackedAxes.ToArray(), lhsLanes.ToArray(), rhsLanes.ToArray(), MaskVectorStyle).FirstOrDefault();
            if (ret is not null)
            {
                return ret;
            }
        }

        return null;
    }
}

[RuleGenerator]
public sealed class WhereUnpackPropagation : RewriteRule<Pattern>
{
    public WhereUnpackPropagation(MaskVectorStyle maskVectorStyle)
    {
        MaskVectorStyle = maskVectorStyle;
    }

    public MaskVectorStyle MaskVectorStyle { get; }

    public override Pattern Pattern { get; } =
    IsWhere(
            "where",
            "caller",
            _ => true,
            IsAlt(
                PatternMatch.F.Tensors.IsUnpack("condUnpack", "cond", _ => true, IsWildcard("condIn")),
                IsWildcard("cond")),
            IsAlt(
                PatternMatch.F.Tensors.IsUnpack("lhsUnpack", "lhs", _ => true, IsWildcard("lhsIn")),
                IsWildcard("lhs")),
            IsAlt(
                PatternMatch.F.Tensors.IsUnpack("rhsUnpack", "rhs", _ => true, IsWildcard("rhsIn")),
                IsWildcard("rhs")));

    public override Expr? GetReplace(IMatchResult result, RunPassContext context)
    {
        var condUnpack = result.GetValueOrDefault("condUnpack") as IR.Tensors.Unpack;
        var lhsUnpack = result.GetValueOrDefault("lhsUnpack") as IR.Tensors.Unpack;
        var rhsUnpack = result.GetValueOrDefault("rhsUnpack") as IR.Tensors.Unpack;
        if ((condUnpack != null && condUnpack.Lanes.Count > 1) || (lhsUnpack != null && lhsUnpack.Lanes.Count > 1) || (rhsUnpack != null && rhsUnpack.Lanes.Count > 1))
        {
            var cond = (Expr)result["cond"];
            var lhs = (Expr)result["lhs"];
            var rhs = (Expr)result["rhs"];
            var candidate = (Expr)result[Pattern];
            var condPackedAxes = (condUnpack != null && condUnpack.Lanes.Count > 1) ? condUnpack.Axes : Array.Empty<int>();
            var lhsPackedAxes = (lhsUnpack != null && lhsUnpack.Lanes.Count > 1) ? lhsUnpack.Axes : Array.Empty<int>();
            var rhsPackedAxes = (rhsUnpack != null && rhsUnpack.Lanes.Count > 1) ? rhsUnpack.Axes : Array.Empty<int>();
            var condLanes = condPackedAxes.Count > 0 ? condUnpack!.Lanes : Array.Empty<int>();
            var lhsLanes = lhsPackedAxes.Count > 0 ? lhsUnpack!.Lanes : Array.Empty<int>();
            var rhsLanes = rhsPackedAxes.Count > 0 ? rhsUnpack!.Lanes : Array.Empty<int>();

            var ret = PackWhere.AddCandidate(cond, lhs, rhs, candidate, condPackedAxes.ToArray(), lhsPackedAxes.ToArray(), rhsPackedAxes.ToArray(), condLanes.ToArray(), lhsLanes.ToArray(), rhsLanes.ToArray(), MaskVectorStyle).FirstOrDefault();
            if (ret is not null)
            {
                return ret;
            }
        }

        return null;
    }
}

[RuleGenerator]
public sealed class GatherUnpackPropagation : RewriteRule<Pattern>
{
    public override Pattern Pattern { get; } =
        IsGather(
            "gather",
            "caller",
            _ => true,
            PatternMatch.F.Tensors.IsUnpack(
                "unpack",
                "callee",
                _ => true,
                IsWildcard("input")),
            IsWildcard("index"));

    public override Expr? GetReplace(IMatchResult result, RunPassContext context)
    {
        var unpack = (IR.Tensors.Unpack)result["unpack"];
        if (unpack.Axes.Count > 1)
        {
            var caller = (Call)result["caller"];
            var callee = (Call)result["callee"];
            var index = (Expr)result["index"];

            var ret = PackGather.AddCandidate(caller, callee, index, unpack.Axes.ToArray(), unpack.Lanes.ToArray()).FirstOrDefault();
            if (ret is not null)
            {
                return ret;
            }
        }

        return null;
    }
}

[RuleGenerator]
public sealed class PackExpandPropagation : RewriteRule<Pattern>
{
    public override Pattern Pattern { get; } =
        PatternMatch.F.Tensors.IsPack(
            "pack",
            "caller",
            _ => true,
            IsExpand(
                "expand",
                "callee",
                _ => true,
                IsWildcard("input", e => e is not Call { Target: IR.Tensors.Unpack }) with { TypePattern = !IsVector() },
                IsFixedShape("shape")));

    public override Expr? GetReplace(IMatchResult result, RunPassContext context)
    {
        var pack = (IR.Tensors.Pack)result["pack"];
        if (pack.Axes.Count > 1)
        {
            var callee = (Call)result["callee"];
            var input = (Expr)result["input"];
            var shape = ((RankedShape)result["shape"]).ToValueArray();

            var ret = PackExpand.AddCandidate(callee, input, shape, pack.Axes.ToArray(), pack.Lanes.ToArray()).FirstOrDefault();
            if (ret is not null)
            {
                return IR.F.Tensors.Pack(ret, pack.Lanes.ToArray(), pack.Axes.ToArray());
            }
        }

        return null;
    }
}

[RuleGenerator]
public sealed class ExpandUnpackPropagation : RewriteRule<Pattern>
{
    public override Pattern Pattern { get; } =
        IsExpand(
            "expand",
            "caller",
            _ => true,
            PatternMatch.F.Tensors.IsUnpack(
                "unpack",
                "callee",
                _ => true,
                IsWildcard("input")),
            IsFixedShape("shape"));

    public override Expr? GetReplace(IMatchResult result, RunPassContext context)
    {
        var unpack = (IR.Tensors.Unpack)result["unpack"];
        if (unpack.Axes.Count > 1)
        {
            var caller = (Call)result["caller"];
            var callee = (Call)result["callee"];
            var shape = ((RankedShape)result["shape"]).ToValueArray();

            var ret = PackExpand.AddCandidate(caller, callee, shape, unpack.Axes.ToArray(), unpack.Lanes.ToArray()).FirstOrDefault();
            if (ret is not null)
            {
                return ret;
            }
        }

        return null;
    }
}
