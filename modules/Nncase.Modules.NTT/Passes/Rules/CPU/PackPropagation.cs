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
        PatternMatch.F.NTT.IsPack(
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
        var pack = (IR.NTT.Pack)result["pack"];
        if (pack.Lanes.Count > 1)
        {
            var op = (IR.Imaging.ResizeImage)result["resize"];
            var input = (Expr)result["input"];
            var newSize = ((TensorConst)result["newSize"]).Value.ToArray<int>();
            var ret = PackResizeImage.AddCandidate(op, input, newSize, pack.Axes.ToArray(), pack.Lanes.ToArray()).FirstOrDefault();
            if (ret is not null)
            {
                return IR.F.NTT.Pack(ret, pack.Lanes.ToArray(), pack.Axes.ToArray());
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
            PatternMatch.F.NTT.IsUnpack(
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
        var unpack = (IR.NTT.Unpack)result["unpack"];
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
        PatternMatch.F.NTT.IsPack(
            "pack",
            "caller",
            _ => true,
            IsReduce(
                "reduce",
                "callee",
                r => r.ReduceOp is ReduceOp.Mean or ReduceOp.Sum,
                IsWildcard("input", e => e is not Call { Target: IR.NTT.Unpack }) with { TypePattern = IsFloat() & !IsVector() },
                IsTensorConst("axes") with { TypePattern = IsIntegral() },
                IsTensorConst("initValue") with { TypePattern = IsFloat() },
                IsTensorConst("keepDims") with { TypePattern = IsBool() }));

    public override Expr? GetReplace(IMatchResult result, RunPassContext context)
    {
        var pack = (IR.NTT.Pack)result["pack"];
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
                return IR.F.NTT.Pack(ret, pack.Lanes.ToArray(), pack.Axes.ToArray());
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
            PatternMatch.F.NTT.IsUnpack(
                "unpack",
                "callee",
                _ => true,
                IsWildcard("input")),
            IsTensorConst("axes") with { TypePattern = IsIntegral() },
            IsTensorConst("initValue") with { TypePattern = IsFloat() },
            IsTensorConst("keepDims") with { TypePattern = IsBool() });

    public override Expr? GetReplace(IMatchResult result, RunPassContext context)
    {
        var unpack = (IR.NTT.Unpack)result["unpack"];
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
public sealed class PackUnaryPropagation : RewriteRule<Pattern>
{
    public override Pattern Pattern { get; } =
        PatternMatch.F.NTT.IsPack(
            "pack",
            "caller",
            _ => true,
            IsUnary(
                "unary",
                "callee",
                _ => true,
                IsWildcard("input", e => e is not Call { Target: IR.NTT.Unpack }) with { TypePattern = IsFloat() & !IsVector() }));

    public override Expr? GetReplace(IMatchResult result, RunPassContext context)
    {
        var pack = (IR.NTT.Pack)result["pack"];
        if (pack.Lanes.Count > 1)
        {
            var op = (IR.Math.Unary)result["unary"];
            var input = (Expr)result["input"];
            var ret = PackUnary.AddCandidate(op, input, pack.Axes.ToArray(), pack.Lanes.ToArray()).FirstOrDefault();
            if (ret is not null)
            {
                return IR.F.NTT.Pack(ret, pack.Lanes.ToArray(), pack.Axes.ToArray());
            }
        }

        return null;
    }
}

[RuleGenerator]
public sealed class UnaryUnpackPropagation : RewriteRule<Pattern>
{
    public override Pattern Pattern { get; } =
        IsUnary(
            "unary",
            "caller",
            _ => true,
            PatternMatch.F.NTT.IsUnpack(
                "unpack",
                "callee",
                _ => true,
                IsWildcard("input")));

    public override Expr? GetReplace(IMatchResult result, RunPassContext context)
    {
        var unpack = (IR.NTT.Unpack)result["unpack"];
        if (unpack.Lanes.Count > 1)
        {
            var op = (IR.Math.Unary)result["unary"];
            var callee = (Expr)result["callee"];
            var ret = PackUnary.AddCandidate(op, callee, unpack.Axes.ToArray(), unpack.Lanes.ToArray()).FirstOrDefault();
            if (ret is not null)
            {
                return ret;
            }
        }

        return null;
    }
}

[RuleGenerator]
public sealed class PackBinaryPropagation : RewriteRule<Pattern>
{
    public override Pattern Pattern { get; } =
        PatternMatch.F.NTT.IsPack(
            "pack",
            "caller",
            _ => true,
            IsBinary(
                "binary",
                "callee",
                _ => true,
                IsWildcard("lhs", e => e is not Call { Target: IR.NTT.Unpack }) with { TypePattern = IsFloat() & !IsVector() },
                IsWildcard("rhs", e => e is not Call { Target: IR.NTT.Unpack }) with { TypePattern = IsFloat() & !IsVector() }));

    public override Expr? GetReplace(IMatchResult result, RunPassContext context)
    {
        var pack = (IR.NTT.Pack)result["pack"];
        if (pack.Lanes.Count > 1)
        {
            var op = (IR.Math.Binary)result["binary"];
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

            var ret = PackBinary.AddCandidate(op, lhs, rhs, candidate, lhsPackedAxes.Select(a => a - lhsExt).ToArray(), rhsPackedAxes.Select(a => a - rhsExt).ToArray(), lhsLanes, rhsLanes).FirstOrDefault();
            if (ret is not null)
            {
                return IR.F.NTT.Pack(ret, pack.Lanes.ToArray(), pack.Axes.ToArray());
            }
        }

        return null;
    }
}

[RuleGenerator]
public sealed class BinaryUnpackPropagation : RewriteRule<Pattern>
{
    public override Pattern Pattern { get; } =
    IsBinary(
            "binary",
            "caller",
            _ => true,
            IsAlt(
                PatternMatch.F.NTT.IsUnpack("lhsUnpack", "lhs", _ => true, IsWildcard("lhsIn")),
                IsWildcard("lhs")),
            IsAlt(
                PatternMatch.F.NTT.IsUnpack("rhsUnpack", "rhs", _ => true, IsWildcard("rhsIn")),
                IsWildcard("rhs")));

    public override Expr? GetReplace(IMatchResult result, RunPassContext context)
    {
        var lhsUnpack = result.GetValueOrDefault("lhsUnpack") == null ? null : (IR.NTT.Unpack)result.GetValueOrDefault("lhsUnpack");
        var rhsUnpack = result.GetValueOrDefault("rhsUnpack") == null ? null : (IR.NTT.Unpack)result.GetValueOrDefault("rhsUnpack");
        if ((lhsUnpack != null && lhsUnpack.Lanes.Count > 1) || (rhsUnpack != null && rhsUnpack.Lanes.Count > 1))
        {
            var op = (IR.Math.Binary)result["binary"];
            var lhs = (Expr)result["lhs"];
            var rhs = (Expr)result["rhs"];
            var candidate = (Expr)result[Pattern];
            var lhsPackedAxes = (lhsUnpack != null && lhsUnpack.Lanes.Count > 1) ? lhsUnpack.Axes : Array.Empty<int>();
            var rhsPackedAxes = (rhsUnpack != null && rhsUnpack.Lanes.Count > 1) ? rhsUnpack.Axes : Array.Empty<int>();
            var lhsLanes = lhsPackedAxes.Count > 0 ? lhsUnpack!.Lanes : Array.Empty<int>();
            var rhsLanes = rhsPackedAxes.Count > 0 ? rhsUnpack!.Lanes : Array.Empty<int>();

            var ret = PackBinary.AddCandidate(op, lhs, rhs, candidate, lhsPackedAxes.ToArray(), rhsPackedAxes.ToArray(), lhsLanes.ToArray(), rhsLanes.ToArray()).FirstOrDefault();
            if (ret is not null)
            {
                return ret;
            }
        }

        return null;
    }
}

[RuleGenerator]
public sealed class PackTransposePropagation : RewriteRule<Pattern>
{
    public override Pattern Pattern { get; } =
        PatternMatch.F.NTT.IsPack(
            "pack",
            "caller",
            _ => true,
            IsTranspose(
                "trans",
                "callee",
                IsWildcard("input", e => e is not Call { Target: IR.NTT.Unpack }) with { TypePattern = IsFloat() & !IsVector() },
                IsTensorConst("perm") with { TypePattern = IsIntegral() }));

    public override Expr? GetReplace(IMatchResult result, RunPassContext context)
    {
        var pack = (IR.NTT.Pack)result["pack"];
        if (pack.Axes.Count > 1)
        {
            var input = (Expr)result["input"];
            var perm = ((TensorConst)result["perm"]).Value.ToArray<int>();
            var permRevsere = Enumerable.Range(0, perm.Length).Select(i => perm.IndexOf(i)).ToArray();
            var packAxes = pack.Axes.Select(a => permRevsere[a]).ToArray();
            var packLanes = Enumerable.Range(0, pack.Lanes.Count).Select(i => pack.Lanes[pack.Axes.IndexOf(packAxes[i])]).ToArray();

            var ret = PackTranspose.AddCandidate(input, perm, packAxes, packLanes).FirstOrDefault();
            if (ret is not null)
            {
                return IR.F.NTT.Pack(ret, pack.Lanes.ToArray(), pack.Axes.ToArray());
            }
        }

        return null;
    }
}

[RuleGenerator]
public sealed class TransposeUnpackPropagation : RewriteRule<Pattern>
{
    public override Pattern Pattern { get; } =
        IsTranspose(
            "trans",
            "caller",
            PatternMatch.F.NTT.IsUnpack(
                "unpack",
                "callee",
                _ => true,
                IsWildcard("input")),
            IsTensorConst("perm") with { TypePattern = IsIntegral() });

    public override Expr? GetReplace(IMatchResult result, RunPassContext context)
    {
        var unpack = (IR.NTT.Unpack)result["unpack"];
        if (unpack.Axes.Count > 1)
        {
            var callee = (Call)result["callee"];
            var perm = ((TensorConst)result["perm"]).Value.ToArray<int>();

            var ret = PackTranspose.AddCandidate(callee, perm, unpack.Axes.ToArray(), unpack.Lanes.ToArray()).FirstOrDefault();
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
        PatternMatch.F.NTT.IsPack(
            "pack",
            "caller",
            _ => true,
            IsUnsqueeze(
                "unsq",
                "callee",
                IsWildcard("input", e => e is not Call { Target: IR.NTT.Unpack }) with { TypePattern = IsFloat() & !IsVector() },
                IsTensorConst("axes") with { TypePattern = IsIntegral() }));

    public override Expr? GetReplace(IMatchResult result, RunPassContext context)
    {
        var pack = (IR.NTT.Pack)result["pack"];
        if (pack.Axes.Count > 1)
        {
            var input = (Expr)result["input"];
            var axes = ((TensorConst)result["axes"]).Value.ToArray<int>();
            var packAxes = pack.Axes.Select(pa => pa - axes.Where(a => a < pa).Count()).ToArray();

            var ret = PackUnsqueeze.AddCandidate(input, axes, packAxes, pack.Lanes.ToArray()).FirstOrDefault();
            if (ret is not null)
            {
                return IR.F.NTT.Pack(ret, pack.Lanes.ToArray(), pack.Axes.ToArray());
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
            PatternMatch.F.NTT.IsUnpack(
                "unpack",
                "callee",
                _ => true,
                IsWildcard("input")),
            IsTensorConst("axes") with { TypePattern = IsIntegral() });

    public override Expr? GetReplace(IMatchResult result, RunPassContext context)
    {
        var unpack = (IR.NTT.Unpack)result["unpack"];
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
public sealed class PackSlicePropagation : RewriteRule<Pattern>
{
    public override Pattern Pattern { get; } =
    PatternMatch.F.NTT.IsPack(
            "pack",
            "caller",
            _ => true,
            IsSlice(
                "slice",
                "callee",
                IsWildcard("input", e => e is not Call { Target: IR.NTT.Unpack }) with { TypePattern = IsFloat() & !IsVector() },
                IsTensorConst("begins") with { TypePattern = IsIntegral() },
                IsTensorConst("ends") with { TypePattern = IsIntegral() },
                IsTensorConst("axes") with { TypePattern = IsIntegral() },
                IsTensorConst("strides") with { TypePattern = IsIntegral() }));

    public override Expr? GetReplace(IMatchResult result, RunPassContext context)
    {
        var pack = (IR.NTT.Pack)result["pack"];
        if (pack.Axes.Count > 1)
        {
            var input = (Expr)result["input"];
            var begins = ((TensorConst)result["begins"]).Value.ToArray<long>();
            var ends = ((TensorConst)result["ends"]).Value.ToArray<long>();
            var axes = ((TensorConst)result["axes"]).Value.ToArray<long>();
            var strides = ((TensorConst)result["strides"]).Value.ToArray<long>();
            var inShape = input.CheckedShape;
            var candidate = (Expr)result[Pattern];
            for (int i = 0; i < axes.Length; i++)
            {
                ends[i] = ends[i] switch
                {
                    < 0 => inShape[axes[i]].FixedValue + ends[i],
                    int.MaxValue => inShape[axes[i]].FixedValue,
                    long.MaxValue => inShape[axes[i]].FixedValue,
                    _ => ends[i],
                };
            }

            if (strides.Any(s => s != 1))
            {
                return null;
            }

            var ret = PackSlice.AddCandidate(input, candidate, begins, ends, axes, strides, pack.Axes.ToArray(), pack.Lanes.ToArray()).FirstOrDefault();
            if (ret is not null)
            {
                return IR.F.NTT.Pack(ret, pack.Lanes.ToArray(), pack.Axes.ToArray());
            }
        }

        return null;
    }
}

[RuleGenerator]
public sealed class SliceUnpackPropagation : RewriteRule<Pattern>
{
    public override Pattern Pattern { get; } =
    IsSlice(
        "slice",
        "caller",
        PatternMatch.F.NTT.IsUnpack(
            "unpack",
            "callee",
            _ => true,
            IsWildcard("input")),
        IsTensorConst("begins") with { TypePattern = IsIntegral() },
        IsTensorConst("ends") with { TypePattern = IsIntegral() },
        IsTensorConst("axes") with { TypePattern = IsIntegral() },
        IsTensorConst("strides") with { TypePattern = IsIntegral() });

    public override Expr? GetReplace(IMatchResult result, RunPassContext context)
    {
        var unpack = (IR.NTT.Unpack)result["unpack"];
        if (unpack.Axes.Count > 1)
        {
            var callee = (Call)result["callee"];
            var begins = ((TensorConst)result["begins"]).Value.ToArray<long>();
            var ends = ((TensorConst)result["ends"]).Value.ToArray<long>();
            var axes = ((TensorConst)result["axes"]).Value.ToArray<long>();
            var strides = ((TensorConst)result["strides"]).Value.ToArray<long>();
            var inShape = callee.CheckedShape;
            var candidate = (Expr)result[Pattern];
            for (int i = 0; i < axes.Length; i++)
            {
                ends[i] = ends[i] switch
                {
                    < 0 => inShape[axes[i]].FixedValue + ends[i],
                    int.MaxValue => inShape[axes[i]].FixedValue,
                    long.MaxValue => inShape[axes[i]].FixedValue,
                    _ => ends[i],
                };
            }

            if (strides.Any(s => s != 1))
            {
                return null;
            }

            var ret = PackSlice.AddCandidate(callee, candidate, begins, ends, axes, strides, unpack.Axes.ToArray(), unpack.Lanes.ToArray()).FirstOrDefault();
            if (ret is not null)
            {
                return ret;
            }
        }

        return null;
    }
}

[RuleGenerator]
public sealed class PackCastPropagation : RewriteRule<Pattern>
{
    public override Pattern Pattern { get; } =
        PatternMatch.F.NTT.IsPack(
            "pack",
            "caller",
            _ => true,
            IsCast(
                "cast",
                "callee",
                _ => true,
                IsWildcard("input", e => e is not Call { Target: IR.NTT.Unpack }) with { TypePattern = IsFloat() & !IsVector() }));

    public override Expr? GetReplace(IMatchResult result, RunPassContext context)
    {
        var pack = (IR.NTT.Pack)result["pack"];
        if (pack.Axes.Count > 1)
        {
            var caller = (Call)result["cast"];
            var input = (Expr)result["input"];
            var candidate = (Expr)result[Pattern];
            var scale = 1f * candidate.CheckedDataType.SizeInBytes / input.CheckedDataType.SizeInBytes;
            var packLanes = pack.Lanes.Select(l => (int)(l * scale)).ToArray();

            var ret = PackCast.AddCandidate(caller, input, pack.Axes.ToArray(), packLanes).FirstOrDefault();
            if (ret is not null)
            {
                return IR.F.NTT.Pack(ret, pack.Lanes.ToArray(), pack.Axes.ToArray());
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
        PatternMatch.F.NTT.IsUnpack(
            "unpack",
            "callee",
            _ => true,
            IsWildcard("input")));

    public override Expr? GetReplace(IMatchResult result, RunPassContext context)
    {
        var unpack = (IR.NTT.Unpack)result["unpack"];
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
