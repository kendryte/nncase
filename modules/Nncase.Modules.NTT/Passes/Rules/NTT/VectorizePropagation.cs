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
public sealed class VectorizeResizeImagePropagation : RewriteRule<Pattern>
{
    public override Pattern Pattern { get; } =
        PatternMatch.F.Tensors.IsVectorize(
            "vectorize",
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
        var vectorize = (IR.Tensors.Vectorize)result["vectorize"];
        if (vectorize.Lanes.Count > 1)
        {
            var op = (IR.Imaging.ResizeImage)result["resize"];
            var input = (Expr)result["input"];
            var newSize = ((TensorConst)result["newSize"]).Value.ToArray<int>();
            var ret = VectorizeResizeImage.AddCandidate(op, input, newSize, vectorize.Axes.ToArray(), vectorize.Lanes.ToArray()).FirstOrDefault();
            if (ret is not null)
            {
                return IR.F.Tensors.Vectorize(ret, vectorize.Lanes.ToArray(), vectorize.Axes.ToArray());
            }
        }

        return null;
    }
}

[RuleGenerator]
public sealed class ResizeDevectorizeImagePropagation : RewriteRule<Pattern>
{
    public override Pattern Pattern { get; } =
        IsResizeImage(
            "resize",
            "caller",
            op => op.TransformationMode == ImageResizeTransformationMode.Asymmetric && op.IsTFResize == false,
            PatternMatch.F.Tensors.IsDevectorize(
                "devectorize",
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
        var devectorize = (IR.Tensors.Devectorize)result["devectorize"];
        if (devectorize.Lanes.Count > 1)
        {
            var op = (IR.Imaging.ResizeImage)result["resize"];
            var callee = (Expr)result["callee"];
            var newSize = ((TensorConst)result["newSize"]).Value.ToArray<int>();
            var ret = VectorizeResizeImage.AddCandidate(op, callee, newSize, devectorize.Axes.ToArray(), devectorize.Lanes.ToArray()).FirstOrDefault();
            if (ret is not null)
            {
                return ret;
            }
        }

        return null;
    }
}

[RuleGenerator]
public sealed class VectorizeReducePropagation : RewriteRule<Pattern>
{
    public override Pattern Pattern { get; } =
        PatternMatch.F.Tensors.IsVectorize(
            "vectorize",
            "caller",
            _ => true,
            IsReduce(
                "reduce",
                "callee",
                r => r.ReduceOp is ReduceOp.Mean or ReduceOp.Sum,
                IsWildcard("input", e => e is not Call { Target: IR.Tensors.Devectorize }) with { TypePattern = IsFloat() & !IsVector() },
                IsTensorConst("axes") with { TypePattern = IsIntegral() },
                IsTensorConst("initValue") with { TypePattern = IsFloat() },
                IsTensorConst("keepDims") with { TypePattern = IsBool() }));

    public override Expr? GetReplace(IMatchResult result, RunPassContext context)
    {
        var vectorize = (IR.Tensors.Vectorize)result["vectorize"];
        if (vectorize.Lanes.Count > 1)
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
            var vectorizeAxes = keepDims ?
                vectorize.Axes.ToArray() :
                vectorize.Axes.Select(a => remainAxes[a]).ToArray();

            var ret = VectorizeReduce.AddCandidate(op, input, axes, initValue, keepDims, vectorizeAxes, vectorize.Lanes.ToArray()).FirstOrDefault();
            if (ret is not null)
            {
                return IR.F.Tensors.Vectorize(ret, vectorize.Lanes.ToArray(), vectorize.Axes.ToArray());
            }
        }

        return null;
    }
}

[RuleGenerator]
public sealed class ReduceDevectorizePropagation : RewriteRule<Pattern>
{
    public override Pattern Pattern { get; } =
        IsReduce(
            "reduce",
            "caller",
            r => r.ReduceOp is ReduceOp.Mean or ReduceOp.Sum,
            PatternMatch.F.Tensors.IsDevectorize(
                "devectorize",
                "callee",
                _ => true,
                IsWildcard("input")),
            IsTensorConst("axes") with { TypePattern = IsIntegral() },
            IsTensorConst("initValue") with { TypePattern = IsFloat() },
            IsTensorConst("keepDims") with { TypePattern = IsBool() });

    public override Expr? GetReplace(IMatchResult result, RunPassContext context)
    {
        var devectorize = (IR.Tensors.Devectorize)result["devectorize"];
        if (devectorize.Lanes.Count > 1)
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
            var ret = VectorizeReduce.AddCandidate(op, callee, axes, initValue, keepDims, devectorize.Axes.ToArray(), devectorize.Lanes.ToArray()).FirstOrDefault();
            if (ret is not null)
            {
                return ret;
            }
        }

        return null;
    }
}

[RuleGenerator]
public sealed class VectorizeUnsqueezePropagation : RewriteRule<Pattern>
{
    public override Pattern Pattern { get; } =
        PatternMatch.F.Tensors.IsVectorize(
            "vectorize",
            "caller",
            _ => true,
            IsUnsqueeze(
                "unsq",
                "callee",
                IsWildcard("input", e => e is not Call { Target: IR.Tensors.Devectorize }) with { TypePattern = IsFloat() & !IsVector() },
                IsTensorConst("axes") with { TypePattern = IsIntegral() }));

    public override Expr? GetReplace(IMatchResult result, RunPassContext context)
    {
        var vectorize = (IR.Tensors.Vectorize)result["vectorize"];
        if (vectorize.Axes.Count > 1)
        {
            var input = (Expr)result["input"];
            var axes = ((TensorConst)result["axes"]).Value.ToArray<int>();
            var vectorizeAxes = vectorize.Axes.Select(pa => pa - axes.Where(a => a < pa).Count()).ToArray();

            var ret = VectorizeUnsqueeze.AddCandidate(input, axes, vectorizeAxes, vectorize.Lanes.ToArray()).FirstOrDefault();
            if (ret is not null)
            {
                return IR.F.Tensors.Vectorize(ret, vectorize.Lanes.ToArray(), vectorize.Axes.ToArray());
            }
        }

        return null;
    }
}

[RuleGenerator]
public sealed class UnsqueezeDevectorizePropagation : RewriteRule<Pattern>
{
    public override Pattern Pattern { get; } =
        IsUnsqueeze(
            "unsq",
            "caller",
            PatternMatch.F.Tensors.IsDevectorize(
                "devectorize",
                "callee",
                _ => true,
                IsWildcard("input")),
            IsTensorConst("axes") with { TypePattern = IsIntegral() });

    public override Expr? GetReplace(IMatchResult result, RunPassContext context)
    {
        var devectorize = (IR.Tensors.Devectorize)result["devectorize"];
        if (devectorize.Axes.Count > 1)
        {
            var calee = (Expr)result["callee"];
            var axes = ((TensorConst)result["axes"]).Value.ToArray<int>();

            var ret = VectorizeUnsqueeze.AddCandidate(calee, axes, devectorize.Axes.ToArray(), devectorize.Lanes.ToArray()).FirstOrDefault();
            if (ret is not null)
            {
                return ret;
            }
        }

        return null;
    }
}

[RuleGenerator]
public sealed class VectorizeCastPropagation : RewriteRule<Pattern>
{
    public override Pattern Pattern { get; } =
        PatternMatch.F.Tensors.IsVectorize(
            "vectorize",
            "caller",
            _ => true,
            IsCast(
                "cast",
                "callee",
                _ => true,
                IsWildcard("input", e => e is not Call { Target: IR.Tensors.Devectorize }) with { TypePattern = IsFloat() & !IsVector() }));

    public override Expr? GetReplace(IMatchResult result, RunPassContext context)
    {
        var vectorize = (IR.Tensors.Vectorize)result["vectorize"];
        if (vectorize.Axes.Count > 1)
        {
            var caller = (Call)result["cast"];
            var input = (Expr)result["input"];
            var candidate = (Expr)result[Pattern];
            var scale = 1f * candidate.CheckedDataType.SizeInBytes / input.CheckedDataType.SizeInBytes;
            var vectorizeLanes = vectorize.Lanes.Select(l => (int)(l * scale)).ToArray();

            var ret = VectorizeCast.AddCandidate(caller, input, vectorize.Axes.ToArray(), vectorizeLanes).FirstOrDefault();
            if (ret is not null)
            {
                return IR.F.Tensors.Vectorize(ret, vectorize.Lanes.ToArray(), vectorize.Axes.ToArray());
            }
        }

        return null;
    }
}

[RuleGenerator]
public sealed class CastDevectorizePropagation : RewriteRule<Pattern>
{
    public override Pattern Pattern { get; } =
    IsCast(
        "cast",
        "caller",
        _ => true,
        PatternMatch.F.Tensors.IsDevectorize(
            "devectorize",
            "callee",
            _ => true,
            IsWildcard("input")));

    public override Expr? GetReplace(IMatchResult result, RunPassContext context)
    {
        var devectorize = (IR.Tensors.Devectorize)result["devectorize"];
        if (devectorize.Axes.Count > 1)
        {
            var caller = (Call)result["caller"];
            var callee = (Expr)result["callee"];

            var ret = VectorizeCast.AddCandidate(caller, callee, devectorize.Axes.ToArray(), devectorize.Lanes.ToArray()).FirstOrDefault();
            if (ret is not null)
            {
                return ret;
            }
        }

        return null;
    }
}

[RuleGenerator]
public sealed class VectorizeComparePropagation : RewriteRule<Pattern>
{
    public VectorizeComparePropagation(MaskVectorStyle maskVectorStyle)
    {
        MaskVectorStyle = maskVectorStyle;
    }

    public MaskVectorStyle MaskVectorStyle { get; }

    public override Pattern Pattern { get; } =
        PatternMatch.F.Tensors.IsVectorize(
            "vectorize",
            "caller",
            _ => true,
            IsCompare(
                "compare",
                "callee",
                _ => true,
                IsWildcard("lhs", e => e is not Call { Target: IR.Tensors.Devectorize }) with { TypePattern = !IsVector() },
                IsWildcard("rhs", e => e is not Call { Target: IR.Tensors.Devectorize }) with { TypePattern = !IsVector() }));

    public override Expr? GetReplace(IMatchResult result, RunPassContext context)
    {
        var vectorize = (IR.Tensors.Vectorize)result["vectorize"];
        if (vectorize.Lanes.Count > 1)
        {
            var op = (IR.Math.Compare)result["compare"];
            var lhs = (Expr)result["lhs"];
            var rhs = (Expr)result["rhs"];
            var lhsShape = lhs.CheckedShape;
            var rhsShape = rhs.CheckedShape;
            var candidate = (Expr)result[Pattern];
            var lhsExt = candidate.CheckedShape.Rank - lhsShape.Rank;
            var rhsExt = candidate.CheckedShape.Rank - rhsShape.Rank;
            var lhsVectorizedAxes = vectorize.Axes.Where(a => a - lhsExt >= 0 && lhsShape[a - lhsExt] is { IsFixed: true } fa && fa != 1).ToArray();
            var rhsVectorizedAxes = vectorize.Axes.Where(a => a - rhsExt >= 0 && rhsShape[a - rhsExt] is { IsFixed: true } fa && fa != 1).ToArray();
            var lhsLanes = lhsVectorizedAxes.Select(a => vectorize.Axes.IndexOf(a)).Select(i => vectorize.Lanes[i]).ToArray();
            var rhsLanes = rhsVectorizedAxes.Select(a => vectorize.Axes.IndexOf(a)).Select(i => vectorize.Lanes[i]).ToArray();

            var ret = VectorizeCompare.AddCandidate(op, lhs, rhs, candidate, lhsVectorizedAxes.Select(a => a - lhsExt).ToArray(), rhsVectorizedAxes.Select(a => a - rhsExt).ToArray(), lhsLanes, rhsLanes, MaskVectorStyle).FirstOrDefault();
            if (ret is not null)
            {
                return IR.F.Tensors.Vectorize(ret, vectorize.Lanes.ToArray(), vectorize.Axes.ToArray());
            }
        }

        return null;
    }
}

[RuleGenerator]
public sealed class CompareDevectorizePropagation : RewriteRule<Pattern>
{
    public CompareDevectorizePropagation(MaskVectorStyle maskVectorStyle)
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
                PatternMatch.F.Tensors.IsDevectorize("lhsDevectorize", "lhs", _ => true, IsWildcard("lhsIn")),
                IsWildcard("lhs")),
            IsAlt(
                PatternMatch.F.Tensors.IsDevectorize("rhsDevectorize", "rhs", _ => true, IsWildcard("rhsIn")),
                IsWildcard("rhs")));

    public override Expr? GetReplace(IMatchResult result, RunPassContext context)
    {
        var lhsDevectorize = result.GetValueOrDefault("lhsDevectorize") as IR.Tensors.Devectorize;
        var rhsDevectorize = result.GetValueOrDefault("rhsDevectorize") as IR.Tensors.Devectorize;
        if ((lhsDevectorize != null && lhsDevectorize.Lanes.Count > 1) || (rhsDevectorize != null && rhsDevectorize.Lanes.Count > 1))
        {
            var op = (IR.Math.Compare)result["compare"];
            var lhs = (Expr)result["lhs"];
            var rhs = (Expr)result["rhs"];
            var candidate = (Expr)result[Pattern];
            var lhsVectorizedAxes = (lhsDevectorize != null && lhsDevectorize.Lanes.Count > 1) ? lhsDevectorize.Axes : Array.Empty<int>();
            var rhsVectorizedAxes = (rhsDevectorize != null && rhsDevectorize.Lanes.Count > 1) ? rhsDevectorize.Axes : Array.Empty<int>();
            var lhsLanes = lhsVectorizedAxes.Count > 0 ? lhsDevectorize!.Lanes : Array.Empty<int>();
            var rhsLanes = rhsVectorizedAxes.Count > 0 ? rhsDevectorize!.Lanes : Array.Empty<int>();

            var ret = VectorizeCompare.AddCandidate(op, lhs, rhs, candidate, lhsVectorizedAxes.ToArray(), rhsVectorizedAxes.ToArray(), lhsLanes.ToArray(), rhsLanes.ToArray(), MaskVectorStyle).FirstOrDefault();
            if (ret is not null)
            {
                return ret;
            }
        }

        return null;
    }
}

[RuleGenerator]
public sealed class WhereDevectorizePropagation : RewriteRule<Pattern>
{
    public WhereDevectorizePropagation(MaskVectorStyle maskVectorStyle)
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
                PatternMatch.F.Tensors.IsDevectorize("condDevectorize", "cond", _ => true, IsWildcard("condIn")),
                IsWildcard("cond")),
            IsAlt(
                PatternMatch.F.Tensors.IsDevectorize("lhsDevectorize", "lhs", _ => true, IsWildcard("lhsIn")),
                IsWildcard("lhs")),
            IsAlt(
                PatternMatch.F.Tensors.IsDevectorize("rhsDevectorize", "rhs", _ => true, IsWildcard("rhsIn")),
                IsWildcard("rhs")));

    public override Expr? GetReplace(IMatchResult result, RunPassContext context)
    {
        var condDevectorize = result.GetValueOrDefault("condDevectorize") as IR.Tensors.Devectorize;
        var lhsDevectorize = result.GetValueOrDefault("lhsDevectorize") as IR.Tensors.Devectorize;
        var rhsDevectorize = result.GetValueOrDefault("rhsDevectorize") as IR.Tensors.Devectorize;
        if ((condDevectorize != null && condDevectorize.Lanes.Count > 1) || (lhsDevectorize != null && lhsDevectorize.Lanes.Count > 1) || (rhsDevectorize != null && rhsDevectorize.Lanes.Count > 1))
        {
            var cond = (Expr)result["cond"];
            var lhs = (Expr)result["lhs"];
            var rhs = (Expr)result["rhs"];
            var candidate = (Expr)result[Pattern];
            var condVectorizedAxes = (condDevectorize != null && condDevectorize.Lanes.Count > 1) ? condDevectorize.Axes : Array.Empty<int>();
            var lhsVectorizedAxes = (lhsDevectorize != null && lhsDevectorize.Lanes.Count > 1) ? lhsDevectorize.Axes : Array.Empty<int>();
            var rhsVectorizedAxes = (rhsDevectorize != null && rhsDevectorize.Lanes.Count > 1) ? rhsDevectorize.Axes : Array.Empty<int>();
            var condLanes = condVectorizedAxes.Count > 0 ? condDevectorize!.Lanes : Array.Empty<int>();
            var lhsLanes = lhsVectorizedAxes.Count > 0 ? lhsDevectorize!.Lanes : Array.Empty<int>();
            var rhsLanes = rhsVectorizedAxes.Count > 0 ? rhsDevectorize!.Lanes : Array.Empty<int>();

            var ret = VectorizeWhere.AddCandidate(cond, lhs, rhs, candidate, condVectorizedAxes.ToArray(), lhsVectorizedAxes.ToArray(), rhsVectorizedAxes.ToArray(), condLanes.ToArray(), lhsLanes.ToArray(), rhsLanes.ToArray(), MaskVectorStyle).FirstOrDefault();
            if (ret is not null)
            {
                return ret;
            }
        }

        return null;
    }
}

[RuleGenerator]
public sealed class GatherDevectorizePropagation : RewriteRule<Pattern>
{
    public override Pattern Pattern { get; } =
        IsGather(
            "gather",
            "caller",
            _ => true,
            PatternMatch.F.Tensors.IsDevectorize(
                "devectorize",
                "callee",
                _ => true,
                IsWildcard("input")),
            IsWildcard("index"));

    public override Expr? GetReplace(IMatchResult result, RunPassContext context)
    {
        var devectorize = (IR.Tensors.Devectorize)result["devectorize"];
        if (devectorize.Axes.Count > 1)
        {
            var caller = (Call)result["caller"];
            var callee = (Call)result["callee"];
            var index = (Expr)result["index"];

            var ret = VectorizeGather.AddCandidate(caller, callee, index, devectorize.Axes.ToArray(), devectorize.Lanes.ToArray()).FirstOrDefault();
            if (ret is not null)
            {
                return ret;
            }
        }

        return null;
    }
}

[RuleGenerator]
public sealed class VectorizeExpandPropagation : RewriteRule<Pattern>
{
    public override Pattern Pattern { get; } =
        PatternMatch.F.Tensors.IsVectorize(
            "vectorize",
            "caller",
            _ => true,
            IsExpand(
                "expand",
                "callee",
                _ => true,
                IsWildcard("input", e => e is not Call { Target: IR.Tensors.Devectorize }) with { TypePattern = !IsVector() },
                IsFixedShape("shape")));

    public override Expr? GetReplace(IMatchResult result, RunPassContext context)
    {
        var vectorize = (IR.Tensors.Vectorize)result["vectorize"];
        if (vectorize.Axes.Count > 1)
        {
            var callee = (Call)result["callee"];
            var input = (Expr)result["input"];
            var shape = ((RankedShape)result["shape"]).ToValueArray();

            var ret = VectorizeExpand.AddCandidate(callee, input, shape, vectorize.Axes.ToArray(), vectorize.Lanes.ToArray()).FirstOrDefault();
            if (ret is not null)
            {
                return IR.F.Tensors.Vectorize(ret, vectorize.Lanes.ToArray(), vectorize.Axes.ToArray());
            }
        }

        return null;
    }
}

[RuleGenerator]
public sealed class ExpandDevectorizePropagation : RewriteRule<Pattern>
{
    public override Pattern Pattern { get; } =
        IsExpand(
            "expand",
            "caller",
            _ => true,
            PatternMatch.F.Tensors.IsDevectorize(
                "devectorize",
                "callee",
                _ => true,
                IsWildcard("input")),
            IsFixedShape("shape"));

    public override Expr? GetReplace(IMatchResult result, RunPassContext context)
    {
        var devectorize = (IR.Tensors.Devectorize)result["devectorize"];
        if (devectorize.Axes.Count > 1)
        {
            var caller = (Call)result["caller"];
            var callee = (Call)result["callee"];
            var shape = ((RankedShape)result["shape"]).ToValueArray();

            var ret = VectorizeExpand.AddCandidate(caller, callee, shape, devectorize.Axes.ToArray(), devectorize.Lanes.ToArray()).FirstOrDefault();
            if (ret is not null)
            {
                return ret;
            }
        }

        return null;
    }
}
