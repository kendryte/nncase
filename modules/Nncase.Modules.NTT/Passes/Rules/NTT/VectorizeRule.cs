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
using Nncase.IR.NN;
using Nncase.IR.Shapes;
using Nncase.IR.Tensors;
using Nncase.PatternMatch;
using Nncase.Utilities;

using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Imaging;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.F.NN;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.NTT;

public abstract class VectorizeRule : RewriteRule<Pattern>
{
    public VectorizeRule(int rank, int lane)
    {
        Rank = rank;
        Lane = lane;
    }

    public int Lane { get; }

    public int Rank { get; }

    public override BaseExpr? GetReplace(IMatchResult result, RunPassContext options) => GetReplaceCandidates(result, options)[0];

    public IEnumerable<int[]> GenerateVectorizeAxes(Shape shape)
    {
        if (shape.IsUnranked || shape.Rank == 0 || (shape.Rank == 1 && shape[0].IsFixed && shape[0].FixedValue == 1))
        {
            yield return Array.Empty<int>();
        }
        else
        {
            yield return Array.Empty<int>();
            for (int i = 0; i < shape.Rank; i++)
            {
                yield return new[] { i };
                for (int j = i + 1; j < shape.Rank; j++)
                {
                    yield return new[] { i, j };
                }
            }
        }
    }

    public (int InitPad, int ExtraPad) FindMinimumPad(int s, int lane, int hierarchy)
    {
        var initPad = (lane - (s % lane)) % lane;
        var pad = 0;

        while ((s + initPad + pad) / lane % hierarchy != 0)
        {
            pad += lane;
        }

        return (initPad, pad);
    }
}

public sealed class VectorizeResizeImage : VectorizeRule
{
    public VectorizeResizeImage(int rank, int lane)
        : base(rank, lane)
    {
    }

    public override Pattern Pattern { get; } = IsResizeImage("target", op => op.TransformationMode == ImageResizeTransformationMode.Asymmetric && op.IsTFResize == false, IsWildcard("input") with { TypePattern = !IsVector() }, IsWildcard("roi"), IsFixedShape("newSize"), IsTensorConst("cubicCoeffA"), IsTensorConst("excludeOutside"), IsTensorConst("extrapolationValue"));

    public static List<Expr> AddCandidate(IR.Imaging.ResizeImage op, Expr input, int[] newSize, int[] vectorizedAxes, int[] lanes)
    {
        var inShape = input.CheckedShape;

        var rets = new List<Expr>();
        if (vectorizedAxes.Length == 0)
        {
            return rets;
        }

        var vectorizedInput = IR.F.Tensors.Pack(VectorizeUtility.PadForVectorize(input, inShape, vectorizedAxes, lanes, 0f, out var padsInput), lanes, vectorizedAxes);

        var resized = IR.F.NTT.ResizeImage(vectorizedInput, new RankedShape(padsInput), vectorizedAxes, newSize, op.ResizeMode, op.TransformationMode, op.NearestMode);

        var post = VectorizeUtility.SliceForVectorize(IR.F.Tensors.Unpack(resized, lanes, vectorizedAxes), inShape, padsInput!);
        if (post.CheckedType is not InvalidType)
        {
            rets.Add(post);
        }

        return rets;
    }

    public override List<Expr> GetReplaceCandidates(IMatchResult result, RunPassContext context)
    {
        var rets = new List<Expr>();
        var roi = (Expr)result["roi"];
        if (roi is not None && ((RankedShape)roi.CheckedShape).Size != 0)
        {
            return rets;
        }

        var op = (IR.Imaging.ResizeImage)result["target"];
        var input = (Expr)result["input"];
        var newSize = ((RankedShape)result["newSize"]).ToValueArray().ToInts();
        var laneSize = Lane / input.CheckedDataType.SizeInBytes;

        rets = AddCandidate(op, input, newSize, [1], [laneSize]);
        return rets;
    }
}

public sealed class VectorizeReduce : VectorizeRule
{
    public VectorizeReduce(int rank, int lane)
        : base(rank, lane)
    {
    }

    public override Pattern Pattern { get; } = IsReduce(
      "target",
      _ => true,
      IsWildcard("input") with { TypePattern = IsFloat() & !IsVector() & HasRankedShape() },
      IsFixedShape("axes"),
      IsTensorConst("initValue") with { TypePattern = IsFloat() },
      IsTensorConst("keepDims") with { TypePattern = IsBool() });

    public static List<Expr> AddCandidate(IR.Math.Reduce op, Expr input, int[] axes, float initValue, bool keepDims, int[] vectorizedAxes, int[] lanes)
    {
        var rets = new List<Expr>();
        var inShape = (RankedShape)input.CheckedShape;
        if (vectorizedAxes.Length == 0)
        {
            return rets;
        }

        var vectorizeReduceAxes = axes.Intersect(vectorizedAxes) == vectorizedAxes;
        if (vectorizeReduceAxes && op.ReduceOp == ReduceOp.Mean)
        {
            return rets;
        }

        axes = axes.Select(x => (int)Util.PositiveIndex(x, inShape.Rank)).ToArray();
        var padValue = vectorizeReduceAxes ? op.ReduceOp switch
        {
            ReduceOp.Mean => 0f,
            ReduceOp.Min => float.MaxValue,
            ReduceOp.Max => float.MinValue,
            ReduceOp.Sum => 0f,
            _ => throw new NotImplementedException(),
        } : 0f;
        var vectorizedInput = IR.F.Tensors.Pack(VectorizeUtility.PadForVectorize(input, inShape, vectorizedAxes, lanes, Tensor.FromScalar(DataTypes.Float32, padValue).CastTo(input.CheckedDataType), out var padsInput), lanes, vectorizedAxes);

        Call reduce = IR.F.NTT.VectorizedReduce(vectorizedInput, op.ReduceOp, axes, initValue, keepDims, vectorizedAxes, new RankedShape(padsInput));

        var (outVectorizeAxes, outPadNums, outLanes, outShape) = IR.NTT.VectorizedReduce.ComputeOutputInfo((IR.NTT.VectorizedReduce)reduce.Target, padsInput, inShape, lanes);
        var post = vectorizeReduceAxes ? reduce : VectorizeUtility.SliceForVectorize(IR.F.Tensors.Unpack(reduce, outLanes, outVectorizeAxes), outShape, outPadNums);

        if (post.CheckedType is not InvalidType)
        {
            rets.Add(post);
        }

        return rets;
    }

    public override IReadOnlyList<BaseExpr> GetReplaceCandidates(IMatchResult result, RunPassContext context)
    {
        var rets = new List<Expr>();
        var op = (IR.Math.Reduce)result["target"];
        var input = (Expr)result["input"];
        var axes = ((RankedShape)result["axes"]).ToValueArray().ToInts();
        if (axes.Length > 1)
        {
            return Array.Empty<BaseExpr>();
        }

        var initValue = ((TensorConst)result["initValue"]).Value.ToScalar<float>();
        var keepDims = ((TensorConst)result["keepDims"]).Value.ToScalar<bool>();
        var laneSize = Lane / input.CheckedDataType.SizeInBytes;
        for (int i = 0; i < input.CheckedShape.Rank; i++)
        {
            rets.AddRange(AddCandidate(op, input, axes, initValue, keepDims, [i], [laneSize]));
            for (int j = i + 1; j < input.CheckedShape.Rank; j++)
            {
                if (Rank > 1)
                {
                    rets.AddRange(AddCandidate(op, input, axes, initValue, keepDims, [i, j], [laneSize, laneSize]));
                }
            }
        }

        return rets;
    }
}

public sealed class VectorizeInstanceNorm : VectorizeRule
{
    public VectorizeInstanceNorm(int rank, int lane)
        : base(rank, lane)
    {
    }

    public override Pattern Pattern { get; } = IsInstanceNormalization(
      "target",
      _ => true,
      IsWildcard("input") with { TypePattern = IsFloat() & !IsVector() },
      IsWildcard("scale") with { TypePattern = IsFloat() & !IsVector() },
      IsWildcard("bias") with { TypePattern = IsFloat() & !IsVector() },
      IsTensorConst("eps") with { TypePattern = IsFloat() });

    public override IReadOnlyList<BaseExpr> GetReplaceCandidates(IMatchResult result, RunPassContext context)
    {
        var rets = new List<Expr>();
        var op = (IR.NN.InstanceNormalization)result["target"];
        var input = (Expr)result["input"];
        var scale = (Expr)result["scale"];
        var bias = (Expr)result["bias"];
        var eps = ((TensorConst)result["eps"]).Value.ToScalar<float>();
        var inShape = input.CheckedShape;
        var pshape = scale.CheckedShape;
        var laneSize = Lane / input.CheckedDataType.SizeInBytes;

        void AddCandidate(int[] vectorizedAxes, int[] lanes)
        {
            var vectorizedInput = IR.F.Tensors.Pack(VectorizeUtility.PadForVectorize(input, inShape, vectorizedAxes, lanes, 0f, out var padsInput), lanes, vectorizedAxes);

            var pAxes = vectorizedAxes.Where(i => i == 1).Select(i => 0).ToArray();
            var vectorizedScale = VectorizeUtility.PadForVectorize(scale, pshape, pAxes, lanes, 0f, out var padsScale);
            if (pAxes.Length > 0)
            {
                vectorizedScale = IR.F.Tensors.Pack(vectorizedScale, Enumerable.Repeat(laneSize, pAxes.Length).ToArray(), pAxes);
            }

            var vectorizedBias = VectorizeUtility.PadForVectorize(bias, pshape, pAxes, lanes, 0f, out var padsBias);
            if (pAxes.Length > 0)
            {
                vectorizedBias = IR.F.Tensors.Pack(vectorizedBias, Enumerable.Repeat(laneSize, pAxes.Length).ToArray(), pAxes);
            }

            var instanceNorm = IR.F.NTT.InstacneNorm(vectorizedInput, vectorizedScale, vectorizedBias, eps, vectorizedAxes, new RankedShape(padsInput));

            if (instanceNorm.CheckedType is not InvalidType)
            {
                var post = VectorizeUtility.SliceForVectorize(IR.F.Tensors.Unpack(instanceNorm, lanes, vectorizedAxes), inShape, padsInput);
                rets.Add(post);
            }
        }

        for (int i = 0; i < input.CheckedShape.Rank; i++)
        {
            AddCandidate(new[] { i }, new[] { laneSize });
        }

        return rets;
    }
}

public sealed class VectorizeMatMul : VectorizeRule
{
    public VectorizeMatMul(int rank = 2, int lane = 16, bool transB = false)
        : base(rank, lane)
    {
        TransB = transB;
    }

    public override Pattern Pattern { get; } = IsMatMul(
      "matmul",
      "target",
      (dytpe) => true,
      IsWildcard("lhs") with { TypePattern = IsFloat() & !IsVector() },
      IsWildcard("rhs") with { TypePattern = IsFloat() & !IsVector() });

    /// <summary>
    /// Gets a value indicating whether trans b, note only for test.
    /// </summary>
    public bool TransB { get; }

    public override IReadOnlyList<BaseExpr> GetReplaceCandidates(IMatchResult result, RunPassContext context)
    {
        var rets = new List<Expr>();
        var lhs = (Expr)result["lhs"];
        var rhs = (Expr)result["rhs"];
        var candidate = (Expr)result[Pattern];
        var lhsShape = lhs.CheckedShape;
        var rhsShape = rhs.CheckedShape;
        var outputDataType = ((Nncase.IR.Math.MatMul)result["matmul"]).OutputDataType;
        var rcontext = new RuleContext(rets, lhs, rhs, candidate, lhsShape, rhsShape, outputDataType);

        // vectorize A's k and B's k
        // AddCandidate(rcontext, VectorizeKind.K, VectorizeKind.K);

        // only vectorize A's m
        // AddCandidate(rcontext, VectorizeKind.M, VectorizeKind.None);

        // only vectorize B's n
        AddCandidate(rcontext, IR.NTT.VectorizedMatMul.VectorizeKind.None, IR.NTT.VectorizedMatMul.VectorizeKind.N/* , transB: rhs is Const */);
        if (Rank > 1)
        {
            // vectorize A's m and B's n, when B is const, force transpose
            AddCandidate(rcontext, IR.NTT.VectorizedMatMul.VectorizeKind.M, IR.NTT.VectorizedMatMul.VectorizeKind.N/* , transB: rhs is Const */);

            // vectorize A's m,k and B's k,n
            AddCandidate(rcontext, IR.NTT.VectorizedMatMul.VectorizeKind.M | IR.NTT.VectorizedMatMul.VectorizeKind.K, IR.NTT.VectorizedMatMul.VectorizeKind.K | IR.NTT.VectorizedMatMul.VectorizeKind.N/* , transB: rhs is Const */);
            if (TransB)
            {
                AddCandidate(rcontext, IR.NTT.VectorizedMatMul.VectorizeKind.M | IR.NTT.VectorizedMatMul.VectorizeKind.K, IR.NTT.VectorizedMatMul.VectorizeKind.K | IR.NTT.VectorizedMatMul.VectorizeKind.N, transB: TransB);
            }

            // vectorize A's m,k and B's k
            // AddCandidate(rcontext,  IR.NTT.VectorizedMatMul.VectorizeKind.M |  IR.NTT.VectorizedMatMul.VectorizeKind.K,  IR.NTT.VectorizedMatMul.VectorizeKind.K);

            // vectorize A's k and B's k,n
            AddCandidate(rcontext, IR.NTT.VectorizedMatMul.VectorizeKind.K, IR.NTT.VectorizedMatMul.VectorizeKind.K | IR.NTT.VectorizedMatMul.VectorizeKind.N/* , transB: lhs is Const */);
        }

        return rets;
    }

    private void AddCandidate(RuleContext context, IR.NTT.VectorizedMatMul.VectorizeKind lhsVectorize, IR.NTT.VectorizedMatMul.VectorizeKind rhsVectorize, bool transA = false, bool transB = false)
    {
        var (rets, lhs, rhs, candidate, _, _, outputDataType) = context;
        var lhsShape = context.LhsShape.ToArray();
        var rhsShape = context.RhsShape.ToArray();
        var lhsLaneSize = Lane / lhs.CheckedDataType.SizeInBytes;
        var rhsLaneSize = Lane / rhs.CheckedDataType.SizeInBytes;
        if (transA)
        {
            var perm = Enumerable.Range(0, lhsShape.Length).ToArray();
            (perm[^2], perm[^1]) = (perm[^1], perm[^2]);
            (lhsShape[^2], lhsShape[^1]) = (lhsShape[^1], lhsShape[^2]);
            lhs = IR.F.Tensors.Transpose(lhs, perm);
        }

        if (transB)
        {
            var perm = Enumerable.Range(0, rhsShape.Length).ToArray();
            (perm[^2], perm[^1]) = (perm[^1], perm[^2]);
            (rhsShape[^2], rhsShape[^1]) = (rhsShape[^1], rhsShape[^2]);
            rhs = IR.F.Tensors.Transpose(rhs, perm);
        }

        int[] lhsLanes;
        int[] lhsVectorizedAxes;
        var (lm, lk) = transA ? (lhsShape.Length - 1, lhsShape.Length - 2) : (lhsShape.Length - 2, lhsShape.Length - 1);
        var (rk, rn) = transB ? (rhsShape.Length - 1, rhsShape.Length - 2) : (rhsShape.Length - 2, rhsShape.Length - 1);
        switch (lhsVectorize)
        {
            case IR.NTT.VectorizedMatMul.VectorizeKind.None:
                lhsLanes = Array.Empty<int>();
                lhsVectorizedAxes = Array.Empty<int>();
                break;
            case IR.NTT.VectorizedMatMul.VectorizeKind.M:
                lhsLanes = [lhsLaneSize];
                lhsVectorizedAxes = [lm];
                break;
            case IR.NTT.VectorizedMatMul.VectorizeKind.K:
                lhsLanes = [lhsLaneSize];
                lhsVectorizedAxes = [lk];
                break;
            case IR.NTT.VectorizedMatMul.VectorizeKind.M | IR.NTT.VectorizedMatMul.VectorizeKind.K:
                lhsLanes = [lhsLaneSize, lhsLaneSize];
                lhsVectorizedAxes = [lm, lk];
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(lhsVectorize), lhsVectorize.ToString());
        }

        int[] rhsLanes;
        int[] rhsVectorizedAxes;
        switch (rhsVectorize)
        {
            case IR.NTT.VectorizedMatMul.VectorizeKind.None:
                rhsLanes = Array.Empty<int>();
                rhsVectorizedAxes = Array.Empty<int>();
                break;
            case IR.NTT.VectorizedMatMul.VectorizeKind.N:
                rhsLanes = [rhsLaneSize];
                rhsVectorizedAxes = [rn];
                break;
            case IR.NTT.VectorizedMatMul.VectorizeKind.K:
                rhsLanes = [rhsLaneSize];
                rhsVectorizedAxes = [rk];
                break;
            case IR.NTT.VectorizedMatMul.VectorizeKind.K | IR.NTT.VectorizedMatMul.VectorizeKind.N:
                rhsLanes = [rhsLaneSize, rhsLaneSize];
                rhsVectorizedAxes = [rk, rn];
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(rhsVectorize), rhsVectorize.ToString());
        }

        var vectorizedLhs = IR.F.Tensors.Pack(VectorizeUtility.PadForVectorize(lhs, lhsShape, lhsVectorizedAxes, lhsLanes, 0f, out var lhsPadNums), lhsLanes, lhsVectorizedAxes);
        var vectorizedRhs = IR.F.Tensors.Pack(VectorizeUtility.PadForVectorize(rhs, rhsShape, rhsVectorizedAxes, rhsLanes, 0f, out var rhsPadNums), rhsLanes, rhsVectorizedAxes);

        var matmul = IR.F.NTT.VectorizedMatMul(vectorizedLhs, vectorizedRhs, lhsVectorizedAxes, rhsVectorizedAxes, transA, transB, false, outputDataType);

        var outRank = System.Math.Max(lhsShape.Length, rhsShape.Length);
        _ = outRank - lhsShape.Length;
        _ = outRank - rhsShape.Length;

        var devectorizeAxes = new List<int>();
        var unpadNums = new List<Dimension>();
        var devectorizeLanes = new List<int>();
        if (lhsVectorize.HasFlag(IR.NTT.VectorizedMatMul.VectorizeKind.M))
        {
            var mVectorizeIndex = Array.IndexOf(lhsVectorizedAxes, lm);
            devectorizeAxes.Add(outRank - 2);
            unpadNums.Add(lhsPadNums[mVectorizeIndex]);
            devectorizeLanes.Add(lhsLaneSize);
        }

        if (rhsVectorize.HasFlag(IR.NTT.VectorizedMatMul.VectorizeKind.N))
        {
            var nVectorizeIndex = Array.IndexOf(rhsVectorizedAxes, rn);
            devectorizeAxes.Add(outRank - 1);
            unpadNums.Add(rhsPadNums[nVectorizeIndex]);
            devectorizeLanes.Add(rhsLaneSize);
        }

        Expr post = matmul;
        if (devectorizeAxes.Any())
        {
            post = VectorizeUtility.SliceForVectorize(IR.F.Tensors.Unpack(matmul, devectorizeLanes.ToArray(), devectorizeAxes.ToArray()), candidate.CheckedShape, unpadNums.ToArray());
        }

        if (post.CheckedType is not InvalidType)
        {
            rets.Add(post);
        }
    }

    private sealed record RuleContext(List<Expr> Results, Expr Lhs, Expr Rhs, Expr Candidate, Shape LhsShape, Shape RhsShape, DataType OutputDataType)
    {
    }
}

public sealed class VectorizeUnary : VectorizeRule
{
    public VectorizeUnary(int rank, int lane)
        : base(rank, lane)
    {
    }

    // FIXME: support exp when rvv exp handles big inputs
    public override Pattern Pattern { get; } = IsUnary(
      "target",
      op => op.UnaryOp is not UnaryOp.Exp,
      IsWildcard("input") with { TypePattern = IsFloat() & !IsVector() });

    public static List<Expr> AddCandidate(IR.Math.Unary op, Expr input, int[] vectorizedAxes, int[] lanes)
    {
        var rets = new List<Expr>();
        var inShape = input.CheckedShape;
        var vectorizedInput = IR.F.Tensors.Pack(VectorizeUtility.PadForVectorize(input, inShape, vectorizedAxes, lanes, 0f, out var padsInput), lanes, vectorizedAxes);

        var unary = IR.F.Math.Unary(op.UnaryOp, vectorizedInput);
        var post = VectorizeUtility.SliceForVectorize(IR.F.Tensors.Unpack(unary, lanes, vectorizedAxes), inShape, padsInput);
        if (post.CheckedType is not InvalidType)
        {
            rets.Add(post);
        }

        return rets;
    }

    public override List<Expr> GetReplaceCandidates(IMatchResult result, RunPassContext context)
    {
        var rets = new List<Expr>();
        var op = (IR.Math.Unary)result["target"];
        var input = (Expr)result["input"];
        var laneSize = Lane / input.CheckedDataType.SizeInBytes;

        for (int i = 0; i < input.CheckedShape.Rank; i++)
        {
            rets.AddRange(AddCandidate(op, input, [i], [laneSize]));
            for (int j = i + 1; j < input.CheckedShape.Rank; j++)
            {
                if (Rank > 1)
                {
                    rets.AddRange(AddCandidate(op, input, [i, j], [laneSize, laneSize]));
                }
            }
        }

        return rets;
    }
}

public sealed class VectorizeBinary : VectorizeRule
{
    public VectorizeBinary(int rank, int lane)
        : base(rank, lane)
    {
    }

    public override Pattern Pattern { get; } = IsBinary(
      "target",
      _ => true,
      IsWildcard("lhs") with { TypePattern = IsFloat() & !IsVector() },
      IsWildcard("rhs") with { TypePattern = IsFloat() & !IsVector() });

    public static List<Expr> AddCandidate(IR.Math.Binary op, Expr lhs, Expr rhs, Expr candidate, int[] lhsVectorizedAxes, int[] rhsVectorizedAxes, int[] lhsLanes, int[] rhsLanes)
    {
        var rets = new List<Expr>();
        var lhsShape = lhs.CheckedShape;
        var rhsShape = rhs.CheckedShape;
        var outShape = candidate.CheckedShape;
        if (lhsVectorizedAxes.Length == 0 && rhsVectorizedAxes.Length == 0)
        {
            return rets;
        }

        var alignedLhsVectorizedAxes = lhsVectorizedAxes.Select(a => a + outShape.Rank - lhsShape.Rank).ToArray();
        var alignedRhsVectorizedAxes = rhsVectorizedAxes.Select(a => a + outShape.Rank - rhsShape.Rank).ToArray();
        if (lhsVectorizedAxes.Any(a => lhsShape[a] is { IsFixed: true, FixedValue: var d } && d == 1)
            || rhsVectorizedAxes.Any(a => rhsShape[a] is { IsFixed: true, FixedValue: var d } && d == 1))
        {
            return rets;
        }

        if (lhsVectorizedAxes.Length > 0 && rhsVectorizedAxes.Length > 0 && !alignedLhsVectorizedAxes.Intersect(alignedRhsVectorizedAxes).Any())
        {
            return rets;
        }

        var alignedLhsShape = Enumerable.Repeat(new DimConst(1), outShape.Rank - lhsShape.Rank).Concat(lhsShape).ToArray();
        var alignedRhsShape = Enumerable.Repeat(new DimConst(1), outShape.Rank - rhsShape.Rank).Concat(rhsShape).ToArray();
        if (alignedLhsVectorizedAxes.Any(a => (alignedRhsShape[a] == alignedLhsShape[a] && !alignedRhsVectorizedAxes.Contains(a)) || (alignedRhsShape[a] != alignedLhsShape[a] && alignedRhsShape[a] != new DimConst(1)))
        || alignedRhsVectorizedAxes.Any(a => (alignedLhsShape[a] == alignedRhsShape[a] && !alignedLhsVectorizedAxes.Contains(a)) || (alignedLhsShape[a] != alignedRhsShape[a] && alignedLhsShape[a] != new DimConst(1))))
        {
            return rets;
        }

        var vectorizedLhs = IR.F.Tensors.Pack(VectorizeUtility.PadForVectorize(lhs, lhsShape, lhsVectorizedAxes, lhsLanes, 0f, out var lhsPadNums), lhsLanes, lhsVectorizedAxes);
        var vectorizedRhs = IR.F.Tensors.Pack(VectorizeUtility.PadForVectorize(rhs, rhsShape, rhsVectorizedAxes, rhsLanes, 0f, out var rhsPadNums), rhsLanes, rhsVectorizedAxes);

        var binary = IR.F.NTT.VectorizedBinary(vectorizedLhs, vectorizedRhs, op.BinaryOp, lhsVectorizedAxes, lhsPadNums, rhsVectorizedAxes, rhsPadNums);
        var post = VectorizeUtility.SliceForVectorize(IR.F.Tensors.Unpack(binary, lhsLanes.Length >= rhsLanes.Length ? lhsLanes : rhsLanes, lhsVectorizedAxes.Length >= rhsVectorizedAxes.Length ? alignedLhsVectorizedAxes : alignedRhsVectorizedAxes), candidate.CheckedShape, lhsVectorizedAxes.Length >= rhsVectorizedAxes.Length ? lhsPadNums! : rhsPadNums!);
        if (post.CheckedType is not InvalidType)
        {
            rets.Add(post);
        }

        return rets;
    }

    public override List<Expr> GetReplaceCandidates(IMatchResult result, RunPassContext context)
    {
        var rets = new List<Expr>();
        var op = (IR.Math.Binary)result["target"];
        var lhs = (Expr)result["lhs"];
        var rhs = (Expr)result["rhs"];
        var candidate = (Expr)result[Pattern];
        var lhsShape = lhs.CheckedShape;
        var rhsShape = rhs.CheckedShape;
        var lhsLaneSize = Lane / lhs.CheckedDataType.SizeInBytes;
        var rhsLaneSize = Lane / rhs.CheckedDataType.SizeInBytes;

        foreach (var arr in new[] { GenerateVectorizeAxes(lhsShape), GenerateVectorizeAxes(rhsShape) }.CartesianProduct())
        {
            var lhsVectorizedAxes = arr.First();
            var rhsVectorizedAxes = arr.Skip(1).First();
            if (lhsVectorizedAxes.Length <= Rank && rhsVectorizedAxes.Length <= Rank)
            {
                rets.AddRange(AddCandidate(op, lhs, rhs, candidate, lhsVectorizedAxes, rhsVectorizedAxes, Enumerable.Repeat(lhsLaneSize, lhsVectorizedAxes.Length).ToArray(), Enumerable.Repeat(rhsLaneSize, rhsVectorizedAxes.Length).ToArray()));
            }
        }

        return rets;
    }
}

public sealed class VectorizeSwish : VectorizeRule
{
    public VectorizeSwish(int rank, int lane)
        : base(rank, lane)
    {
    }

    public override Pattern Pattern { get; } = IsSwish(
      "target",
      IsWildcard("input") with { TypePattern = IsFloat() & !IsVector() },
      IsTensorConst("beta") with { TypePattern = IsFloatScalar() });

    public override List<Expr> GetReplaceCandidates(IMatchResult result, RunPassContext context)
    {
        var rets = new List<Expr>();
        var input = (Expr)result["input"];
        var beta = ((TensorConst)result["beta"]).Value.ToScalar<float>();
        var inShape = input.CheckedShape;
        var laneSize = Lane / input.CheckedDataType.SizeInBytes;

        void AddCandidate(int[] vectorizedAxes, int[] lanes)
        {
            var vectorized = IR.F.Tensors.Pack(VectorizeUtility.PadForVectorize(input, inShape, vectorizedAxes, lanes, 0f, out var pads), lanes, vectorizedAxes);

            var swish = IR.F.NN.Swish(vectorized, beta);
            var post = VectorizeUtility.SliceForVectorize(IR.F.Tensors.Unpack(swish, lanes, vectorizedAxes), inShape, pads);
            if (post.CheckedType is not InvalidType)
            {
                rets.Add(post);
            }
        }

        for (int i = 0; i < input.CheckedShape.Rank; i++)
        {
            AddCandidate(new[] { i }, new[] { laneSize });
            for (int j = i + 1; j < input.CheckedShape.Rank; j++)
            {
                if (Rank > 1)
                {
                    AddCandidate(new[] { i, j }, new[] { laneSize, laneSize });
                }
            }
        }

        return rets;
    }
}

public sealed class VectorizeTranspose : VectorizeRule
{
    public VectorizeTranspose(int rank = 2, int lane = 16)
        : base(rank, lane)
    {
    }

    public override Pattern Pattern { get; } = IsTranspose(
      "trans",
      IsWildcard("input") with { TypePattern = IsFloat() & !IsVector() },
      IsFixedShape("perm"));

    public static List<Expr> AddCandidate(Expr input, int[] perm, int[] vectorizedAxes, int[] lanes)
    {
        var rets = new List<Expr>();
        var inShape = input.CheckedShape;
        var vectorized = IR.F.Tensors.Pack(VectorizeUtility.PadForVectorize(input, inShape, vectorizedAxes, lanes, 0f, out var pads), lanes, vectorizedAxes);

        var tarns = IR.F.Tensors.Transpose(vectorized, perm);
        if (tarns.CheckedType is not InvalidType)
        {
            var partialPerm = perm.Select(axis => vectorizedAxes.IndexOf(axis)).Where(x => x != -1).ToArray();
            var devectorizeAxes = vectorizedAxes.Select(axis => perm.IndexOf(axis)).ToArray();
            var devectorizePads = Enumerable.Range(0, pads.Length).Select(i => pads[partialPerm[i]]).ToArray();
            var devectorizeLanes = Enumerable.Range(0, lanes.Length).Select(i => lanes[partialPerm[i]]).ToArray();
            var newShape = perm.Select(i => inShape[i]).ToArray();
            var post = VectorizeUtility.SliceForVectorize(IR.F.Tensors.Unpack(tarns, devectorizeLanes, devectorizeAxes), newShape, devectorizePads);
            if (post.CheckedType is not InvalidType)
            {
                rets.Add(post);
            }
        }

        return rets;
    }

    public override List<Expr> GetReplaceCandidates(IMatchResult result, RunPassContext context)
    {
        var rets = new List<Expr>();

        var input = (Expr)result["input"];
        var perm = ((RankedShape)result["perm"]).ToValueArray().ToInts();
        var laneSize = Lane / input.CheckedDataType.SizeInBytes;

        for (int i = 0; i < input.CheckedShape.Rank; i++)
        {
            rets.AddRange(AddCandidate(input, perm, [i], [laneSize]));
            for (int j = i + 1; j < input.CheckedShape.Rank; j++)
            {
                if (Rank > 1)
                {
                    rets.AddRange(AddCandidate(input, perm, [i, j], [laneSize, laneSize]));
                }
            }
        }

        return rets;
    }
}

public sealed class VectorizeUnsqueeze : VectorizeRule
{
    public VectorizeUnsqueeze(int rank, int lane)
        : base(rank, lane)
    {
    }

    public override Pattern Pattern { get; } = IsUnsqueeze(
      "unsq",
      IsWildcard("input") with { TypePattern = IsFloat() & !IsVector() },
      IsFixedShape("axes"));

    public static List<Expr> AddCandidate(Expr input, int[] axes, int[] vectorizedAxes, int[] lanes)
    {
        var rets = new List<Expr>();
        var inShape = input.CheckedShape;
        if (vectorizedAxes.Length == 0)
        {
            return rets;
        }

        var vectorized = IR.F.Tensors.Pack(VectorizeUtility.PadForVectorize(input, inShape, vectorizedAxes, lanes, 0f, out var pads), lanes, vectorizedAxes);

        var unseq = IR.F.Tensors.Unsqueeze(vectorized, axes);
        var devectorizeAxes = vectorizedAxes.Select(axis => axis + axes.Count(i => i <= axis)).ToArray();
        var outShape = inShape.ToList();
        foreach (var axis in axes)
        {
            if (axis >= 0)
            {
                outShape.Insert(axis, 1);
            }
            else
            {
                var index = System.Math.Max(outShape.Count + axis + 1, 0);
                outShape.Insert(index, 1);
            }
        }

        var post = VectorizeUtility.SliceForVectorize(IR.F.Tensors.Unpack(unseq, lanes, devectorizeAxes), outShape.ToArray(), pads!);
        if (post.CheckedType is not InvalidType)
        {
            rets.Add(post);
        }

        return rets;
    }

    public override List<Expr> GetReplaceCandidates(IMatchResult result, RunPassContext context)
    {
        var rets = new List<Expr>();

        var input = (Expr)result["input"];
        var axes = ((RankedShape)result["axes"]).ToValueArray().ToInts();
        var laneSize = Lane / input.CheckedDataType.SizeInBytes;

        for (int i = 0; i < input.CheckedShape.Rank; i++)
        {
            rets.AddRange(AddCandidate(input, axes, [i], [laneSize]));
            for (int j = i + 1; j < input.CheckedShape.Rank; j++)
            {
                if (Rank > 1)
                {
                    rets.AddRange(AddCandidate(input, axes, [i, j], [laneSize, laneSize]));
                }
            }
        }

        return rets;
    }
}

public sealed class VectorizeConv2D : VectorizeRule
{
    public VectorizeConv2D(int rank, int lane)
        : base(rank, lane)
    {
    }

    public override Pattern Pattern { get; } = IsConv2D(
        "conv",
        conv => conv.PadMode == PadMode.Constant,
        IsWildcard("input") with { TypePattern = !IsVector() },
        IsWildcard("weights"),
        IsWildcard("bias"),
        IsFixedShape("stride"),
        IsFixedPaddings("padding"),
        IsFixedShape("dilation"),
        IsFixedDimension("groups"),
        IsTensorConst("fusedClamp"));

    public static Expr AddCandidate(Expr input, Expr weights, Expr bias, int[] strides, int[] padding, long[] wShape, long[] outShape)
    {
        var col = IR.F.NTT.Im2col(input, new[] { wShape[2], wShape[3] }, strides, padding);
        var newW = IR.F.Tensors.Reshape(weights, new[] { wShape[0], wShape[1] * wShape[2] * wShape[3] });
        var matmul = IR.F.Tensors.MatMul(newW, col); // [oc, b*oh*ow]
        var newBias = IR.F.Tensors.Reshape(bias, new[] { wShape[0], 1 });
        var add = matmul + newBias;
        if (outShape[0] == 1)
        {
            return IR.F.Tensors.Reshape(add, outShape);
        }

        return IR.F.Tensors.Transpose(IR.F.Tensors.Reshape(add, new[] { outShape[1], outShape[0], outShape[2], outShape[3] }), new[] { 1, 0, 2, 3 });
    }

    public static Expr AddVectorizedCandidate(Expr input, Expr weights, Expr bias, int[] strides, int[] padding, long[] wShape, long[] outShape, int lane)
    {
        var paddedInput = VectorizeUtility.PadForVectorize(input, input.CheckedShape.ToValueArray(), new[] { 1 }, new[] { lane }, 0f, out _);
        var col = IR.F.NTT.Im2col(IR.F.Tensors.Pack(paddedInput, new[] { lane }, new[] { 1 }), new[] { wShape[2], wShape[3] }, strides, padding, new[] { 1 }, new[] { 0 });
        var paddedW = VectorizeUtility.PadForVectorize(weights, wShape, new[] { 1 }, new[] { lane }, 0f, out _);
        var newW = IR.F.Tensors.Reshape(IR.F.Tensors.Pack(paddedW, new[] { lane }, new[] { 1 }), new[] { wShape[0], MathUtility.CeilDiv(wShape[1], lane) * wShape[2] * wShape[3] });
        var matmul = IR.F.NTT.VectorizedMatMul(newW, col, new[] { 1 }, new[] { 0 }, false, false, false); // [oc, b*oh*ow]
        var newBias = IR.F.Tensors.Reshape(bias, new[] { wShape[0], 1 });
        var add = matmul + newBias;
        if (outShape[0] == 1)
        {
            return IR.F.Tensors.Reshape(add, outShape);
        }
        else
        {
            return IR.F.Tensors.Transpose(IR.F.Tensors.Reshape(add, new[] { outShape[1], outShape[0], outShape[2], outShape[3] }), new[] { 1, 0, 2, 3 });
        }
    }

    public override IReadOnlyList<BaseExpr> GetReplaceCandidates(IMatchResult result, RunPassContext context)
    {
        var rets = new List<Expr>();

        var input = (Expr)result["input"];
        var weights = (Expr)result["weights"];
        var bias = (Expr)result["bias"];
        var strides = ((RankedShape)result["stride"]).ToValueArray().ToInts();
        var padding = Tensor.From(((Paddings)result["padding"]).ToValueArray()).ToArray<int>();
        var dilation = ((RankedShape)result["dilation"]).ToValueArray().ToInts();
        var groups = (int)((DimConst)result["groups"]).Value;
        var fusedClamp = ((TensorConst)result["fusedClamp"]).Value.ToArray<float>();
        var wShape = weights.CheckedShape.ToValueArray();
        var outShape = ((Expr)result[Pattern]).CheckedShape.ToValueArray();
        var laneSize = Lane / input.CheckedDataType.SizeInBytes;
        if (groups != 1 || wShape[1] % Lane != 0 || dilation[0] != 1 || dilation[1] != 1 || fusedClamp[0] != float.NegativeInfinity || fusedClamp[1] != float.PositiveInfinity)
        {
            return rets;
        }

        // only vectorize on in channels
        rets.Add(AddCandidate(input, weights, bias, strides, padding, wShape, outShape));
        rets.Add(AddVectorizedCandidate(input, weights, bias, strides, padding, wShape, outShape, laneSize));
        return rets;
    }
}

public sealed class VectorizeReshape : VectorizeRule
{
    public VectorizeReshape(int rank, int lane)
        : base(rank, lane)
    {
    }

    public override Pattern Pattern { get; } = IsReshape(
      "target",
      IsWildcard("input") with { TypePattern = !IsVector() },
      IsRankedShape("newShape"));

    public static List<Expr> AddCandidate(Expr input, RankedShape newShape, Dictionary<int, List<int>> forwardDict, Dictionary<int, List<int>> backwardDict, int[] vectorizedAxes, int[] lanes)
    {
        var rets = new List<Expr>();
        var inShape = input.CheckedShape;
        if (vectorizedAxes.Length == 0)
        {
            return rets;
        }

        // not support vectorize on dynamic dims.
        if (inShape.Any(s => s is { IsFixed: false } && vectorizedAxes.Contains(inShape.ToArray().IndexOf(s))))
        {
            return rets;
        }

        // 1. skip when the vectorizedAxes will be split or merge.
        var devectorizeAxes = new List<int>();
        foreach (var axis in vectorizedAxes)
        {
            var mapedOutAxes = forwardDict[axis];
            if (mapedOutAxes.Count > 1)
            {
                if (mapedOutAxes.Count(i => newShape[i].FixedValue != 1) > 1)
                {
                    // we can vectorize on split axis and devectorize on splited last axis.
                    devectorizeAxes.Add(mapedOutAxes[^1]);
                }
                else
                {
                    // unsqueeze.
                    var outAxis = mapedOutAxes.FirstOrDefault(i => newShape[i].FixedValue != 1, mapedOutAxes.First());
                    if (backwardDict[outAxis].Count != 1)
                    {
                        continue;
                    }

                    devectorizeAxes.Add(outAxis);
                }
            }
            else
            {
                var outAxis = mapedOutAxes.First();

                // when the outAxis is merged dim, only support no transpose order and no pad.
                var inAxes = backwardDict[outAxis];
                if (inAxes.Count == 1 || (inAxes[^1] == axis && inShape[axis] % lanes[vectorizedAxes.IndexOf(axis)] == 0))
                {
                    devectorizeAxes.Add(outAxis);
                }
                else
                {
                    return rets;
                }
            }
        }

        if (devectorizeAxes.Count == 0 || devectorizeAxes.Count != vectorizedAxes.Length)
        {
            return rets;
        }

        var vectorized = IR.F.Tensors.Pack(VectorizeUtility.PadForVectorize(input, inShape, vectorizedAxes, lanes, 0f, out var pads), lanes, vectorizedAxes);

        var vectorizedNewShape = newShape.ToArray();
        foreach (var (lane, axis) in lanes.Zip(devectorizeAxes))
        {
            vectorizedNewShape[axis] = Dimension.CeilDiv(vectorizedNewShape[axis], lane);
        }

        var nreshape = IR.F.Tensors.Reshape(vectorized, vectorizedNewShape);
        var post = VectorizeUtility.SliceForVectorize(IR.F.Tensors.Unpack(nreshape, lanes, devectorizeAxes.ToArray()), newShape, pads!);
        if (post.CheckedType is not InvalidType)
        {
            rets.Add(post);
        }

        return rets;
    }

    public override List<Expr> GetReplaceCandidates(IMatchResult result, RunPassContext context)
    {
        var rets = new List<Expr>();

        var input = (Expr)result["input"];
        var newShape = (RankedShape)result["newShape"];
        var inShape = input.CheckedShape;
        var laneSize = Lane / input.CheckedDataType.SizeInBytes;

        // 1. find the mapping transforms
        if (!IRUtility.TryGetShapeMapMatrix(CompilerServices.GetMaxShape(inShape), CompilerServices.GetMaxShape(newShape), out var mat))
        {
            return new List<Expr> { };
        }

        var (forwardDict, backwardDict) = IRUtility.ShapeMapMatrixAsDict(mat);

        for (int i = 0; i < input.CheckedShape.Rank; i++)
        {
            rets.AddRange(AddCandidate(input, newShape, forwardDict, backwardDict, new[] { i }, new[] { laneSize }));
            if (Rank > 1)
            {
                for (int j = i + 1; j < input.CheckedShape.Rank; j++)
                {
                    rets.AddRange(AddCandidate(input, newShape, forwardDict, backwardDict, new[] { i, j }, new[] { laneSize, laneSize }));
                }
            }
        }

        return rets;
    }
}

public sealed class VectorizeSlice : VectorizeRule
{
    public VectorizeSlice(int rank, int lane)
        : base(rank, lane)
    {
    }

    public override Pattern Pattern { get; } = IsSlice(
      "target",
      IsWildcard("input") with { TypePattern = IsFloat() & !IsVector() },
      IsFixedShape("begins"),
      IsFixedShape("ends"),
      IsFixedShape("axes"),
      IsFixedShape("strides"));

    public static List<Expr> AddCandidate(Expr input, Expr candidate, long[] begins, long[] ends, long[] axes, long[] strides, int[] vectorizeAxes, int[] lanes)
    {
        var rets = new List<Expr>();
        if (vectorizeAxes.Length == 0)
        {
            return rets;
        }

        var inShape = input.CheckedShape;
        var vectorizedBegins = begins.ToArray();
        var vectorizedEnds = ends.ToArray();
        for (int i = 0; i < vectorizeAxes.Length; i++)
        {
            var vectorizeAxis = vectorizeAxes[i];
            int j = axes.IndexOf(vectorizeAxis);

            // when the slice axis was vectorized, it must have no pad.
            if (j != -1)
            {
                if (begins[j] % lanes[i] == 0 && ends[j] % lanes[i] == 0)
                {
                    vectorizedBegins[j] = begins[j] / lanes[i];
                    vectorizedEnds[j] = ends[j] / lanes[i];
                }
                else
                {
                    return rets;
                }
            }
        }

        var vectorized = IR.F.Tensors.Pack(VectorizeUtility.PadForVectorize(input, inShape, vectorizeAxes, lanes, 0f, out var pads), lanes, vectorizeAxes);

        var slice = IR.F.Tensors.Slice(vectorized, vectorizedBegins, vectorizedEnds, axes, strides);
        var post = VectorizeUtility.SliceForVectorize(IR.F.Tensors.Unpack(slice, lanes, vectorizeAxes), candidate.CheckedShape, pads!);
        if (post.CheckedType is not InvalidType)
        {
            rets.Add(post);
        }

        return rets;
    }

    public override List<Expr> GetReplaceCandidates(IMatchResult result, RunPassContext context)
    {
        var rets = new List<Expr>();

        var input = (Expr)result["input"];
        var begins = ((RankedShape)result["begins"]).ToValueArray();
        var ends = ((RankedShape)result["ends"]).ToValueArray();
        var axes = ((RankedShape)result["axes"]).ToValueArray();
        var strides = ((RankedShape)result["strides"]).ToValueArray();
        var inShape = input.CheckedShape;
        var candidate = (Expr)result[Pattern];
        var laneSize = Lane / input.CheckedDataType.SizeInBytes;
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
            return rets;
        }

        for (int i = 0; i < input.CheckedShape.Rank; i++)
        {
            rets.AddRange(AddCandidate(input, candidate, begins, ends, axes, strides, [i], [laneSize]));
            for (int j = i + 1; j < input.CheckedShape.Rank; j++)
            {
                if (Rank > 1)
                {
                    rets.AddRange(AddCandidate(input, candidate, begins, ends, axes, strides, new[] { i, j }, new[] { laneSize, laneSize }));
                }
            }
        }

        return rets;
    }
}

public sealed class VectorizeConcat : VectorizeRule
{
    public VectorizeConcat(int rank, int lane)
        : base(rank, lane)
    {
    }

    public override Pattern Pattern { get; } = IsConcat(
        "concat",
        "concatCall",
        _ => true,
        IsTuple(null, IsVArgsRepeat("tupleInputs", exprs =>
        {
            var patterns = new Pattern[exprs.Length];
            for (var i = 0; i < patterns.Length; i++)
            {
                patterns[i] = IsWildcard($"input_{i}") with { TypePattern = IsFloat() & !IsVector() };
            }

            return patterns;
        })));

    public static List<Expr> AddCandidate(BaseExpr[] inputs, Expr candidate, int axis, int[] vectorizeAxes, int[] lanes)
    {
        var rets = new List<Expr>();
        if (vectorizeAxes.Length == 0)
        {
            return rets;
        }

        var vectorizedInputs = new Expr[inputs.Length];
        Dimension[]? pads = null;
        for (var i = 0; i < inputs.Length; i++)
        {
            var inShape = inputs[i].CheckedShape;
            vectorizedInputs[i] = IR.F.Tensors.Pack(VectorizeUtility.PadForVectorize((Expr)inputs[i], inShape, vectorizeAxes, lanes, 0f, out pads), lanes, vectorizeAxes);
        }

        var concat = IR.F.Tensors.Concat(new IR.Tuple(vectorizedInputs), axis);
        var post = VectorizeUtility.SliceForVectorize(IR.F.Tensors.Unpack(concat, lanes, vectorizeAxes), candidate.CheckedShape, pads!);
        if (post.CheckedType is not InvalidType)
        {
            rets.Add(post);
        }

        return rets;
    }

    public override List<Expr> GetReplaceCandidates(IMatchResult result, RunPassContext context)
    {
        var rets = new List<Expr>();

        var tupleInputs = (IReadOnlyList<BaseExpr>)result["tupleInputs"];
        var op = (IR.Tensors.Concat)result["concat"];
        var axis = op.Axis;
        var candidate = (Expr)result[Pattern];
        var laneSize = Lane / tupleInputs[0].CheckedDataType.SizeInBytes;

        for (int i = 0; i < tupleInputs[0].CheckedShape.Rank; i++)
        {
            rets.AddRange(AddCandidate(tupleInputs.ToArray(), candidate, axis, [i], [laneSize]));
        }

        return rets;
    }
}

public sealed class VectorizeCompare : VectorizeRule
{
    public VectorizeCompare(MaskVectorStyle maskVectorStyle, int rank, int lane)
        : base(rank, lane)
    {
        MaskVectorStyle = maskVectorStyle;
    }

    public MaskVectorStyle MaskVectorStyle { get; }

    public override Pattern Pattern { get; } = IsCompare(
      "target",
      _ => true,
      IsWildcard("lhs") with { TypePattern = IsFloat() & !IsVector() },
      IsWildcard("rhs") with { TypePattern = IsFloat() & !IsVector() });

    public static List<Expr> AddCandidate(IR.Math.Compare op, Expr lhs, Expr rhs, Expr candidate, int[] lhsVectorizedAxes, int[] rhsVectorizedAxes, int[] lhsLanes, int[] rhsLanes, MaskVectorStyle maskVectorStyle)
    {
        var rets = new List<Expr>();
        var lhsShape = lhs.CheckedShape;
        var rhsShape = rhs.CheckedShape;
        var outShape = candidate.CheckedShape;
        if (lhsVectorizedAxes.Length == 0 && rhsVectorizedAxes.Length == 0)
        {
            return rets;
        }

        var alignedLhsVectorizedAxes = lhsVectorizedAxes.Select(a => a + outShape.Rank - lhsShape.Rank).ToArray();
        var alignedRhsVectorizedAxes = rhsVectorizedAxes.Select(a => a + outShape.Rank - rhsShape.Rank).ToArray();
        if (lhsVectorizedAxes.Any(a => lhsShape[a] is { IsFixed: true, FixedValue: var d } && d == 1)
        || rhsVectorizedAxes.Any(a => rhsShape[a] is { IsFixed: true, FixedValue: var d } && d == 1))
        {
            return rets;
        }

        if (lhsVectorizedAxes.Length > 0 && rhsVectorizedAxes.Length > 0 && !alignedLhsVectorizedAxes.Intersect(alignedRhsVectorizedAxes).Any())
        {
            return rets;
        }

        var alignedLhsShape = Enumerable.Repeat(new DimConst(1), outShape.Rank - lhsShape.Rank).Concat(lhsShape).ToArray();
        var alignedRhsShape = Enumerable.Repeat(new DimConst(1), outShape.Rank - rhsShape.Rank).Concat(rhsShape).ToArray();
        if (alignedLhsVectorizedAxes.Any(a => (alignedRhsShape[a] == alignedLhsShape[a] && !alignedRhsVectorizedAxes.Contains(a)) || (alignedRhsShape[a] != alignedLhsShape[a] && alignedRhsShape[a] != new DimConst(1)))
        || alignedRhsVectorizedAxes.Any(a => (alignedLhsShape[a] == alignedRhsShape[a] && !alignedLhsVectorizedAxes.Contains(a)) || (alignedLhsShape[a] != alignedRhsShape[a] && alignedLhsShape[a] != new DimConst(1))))
        {
            return rets;
        }

        var vectorizedLhs = IR.F.Tensors.Pack(VectorizeUtility.PadForVectorize(lhs, lhsShape, lhsVectorizedAxes, lhsLanes, 0f, out var lhsPadNums), lhsLanes, lhsVectorizedAxes);
        var vectorizedRhs = IR.F.Tensors.Pack(VectorizeUtility.PadForVectorize(rhs, rhsShape, rhsVectorizedAxes, rhsLanes, 0f, out var rhsPadNums), rhsLanes, rhsVectorizedAxes);

        var compare = IR.F.Math.Compare(op.CompareOp, vectorizedLhs, vectorizedRhs, maskVectorStyle);
        var post = VectorizeUtility.SliceForVectorize(IR.F.Tensors.Unpack(compare, lhsLanes.Length >= rhsLanes.Length ? lhsLanes : rhsLanes, lhsVectorizedAxes.Length >= rhsVectorizedAxes.Length ? alignedLhsVectorizedAxes : alignedRhsVectorizedAxes), candidate.CheckedShape, lhsVectorizedAxes.Length >= rhsVectorizedAxes.Length ? lhsPadNums! : rhsPadNums!);
        if (post.CheckedType is not InvalidType)
        {
            rets.Add(post);
        }

        return rets;
    }

    public override List<Expr> GetReplaceCandidates(IMatchResult result, RunPassContext context)
    {
        var rets = new List<Expr>();
        var op = (IR.Math.Compare)result["target"];
        var lhs = (Expr)result["lhs"];
        var rhs = (Expr)result["rhs"];
        var candidate = (Expr)result[Pattern];
        var lhsShape = lhs.CheckedShape;
        var rhsShape = rhs.CheckedShape;
        var lhsLaneSize = Lane / lhs.CheckedDataType.SizeInBytes;
        var rhsLaneSize = Lane / rhs.CheckedDataType.SizeInBytes;

        foreach (var arr in new[] { GenerateVectorizeAxes(lhsShape), GenerateVectorizeAxes(rhsShape) }.CartesianProduct())
        {
            var lhsVectorizedAxes = arr.First();
            var rhsVectorizedAxes = arr.Skip(1).First();
            if (lhsVectorizedAxes.Length <= Rank && rhsVectorizedAxes.Length <= Rank)
            {
                rets.AddRange(AddCandidate(op, lhs, rhs, candidate, lhsVectorizedAxes, rhsVectorizedAxes, Enumerable.Repeat(lhsLaneSize, lhsVectorizedAxes.Length).ToArray(), Enumerable.Repeat(rhsLaneSize, rhsVectorizedAxes.Length).ToArray(), MaskVectorStyle));
            }
        }

        return rets;
    }
}

public sealed class VectorizeCast : VectorizeRule
{
    public VectorizeCast(int rank, int lane)
        : base(rank, lane)
    {
    }

    public override Pattern Pattern { get; } = IsCast(
      "target",
      "call",
      _ => true,
      IsWildcard("input") with { TypePattern = IsFloat() & !IsVector() });

    public static List<Expr> AddCandidate(Call call, Expr input, int[] vectorizedAxes, int[] lanes)
    {
        var rets = new List<Expr>();
        var op = (IR.Tensors.Cast)call.Target;
        var inShape = input.CheckedShape;
        if (vectorizedAxes.Length == 0)
        {
            return rets;
        }

        var vectorizedInput = IR.F.Tensors.Pack(VectorizeUtility.PadForVectorize(input, inShape, vectorizedAxes, lanes, 0f, out var padsInput), lanes, vectorizedAxes);

        var scale = 1f * call.CheckedDataType.SizeInBytes / input.CheckedDataType.SizeInBytes;
        var outLanes = lanes.Select(l => (int)(l / scale)).ToArray();
        var newType = new VectorType(op.NewType, outLanes);

        var cast = IR.F.Tensors.Cast(vectorizedInput, newType, op.CastMode, vectorizedAxes);
        var post = VectorizeUtility.SliceForVectorize(IR.F.Tensors.Unpack(cast, outLanes, vectorizedAxes), inShape, padsInput!);
        if (cast.CheckedType is not InvalidType)
        {
            rets.Add(post);
        }

        return rets;
    }

    public override List<Expr> GetReplaceCandidates(IMatchResult result, RunPassContext context)
    {
        var rets = new List<Expr>();
        var call = (Call)result["call"];
        var input = (Expr)result["input"];
        var laneSize = Lane / input.CheckedDataType.SizeInBytes;

        for (int i = 0; i < input.CheckedShape.Rank; i++)
        {
            rets.AddRange(AddCandidate(call, input, [i], [laneSize]));
            for (int j = i + 1; j < input.CheckedShape.Rank; j++)
            {
                if (Rank > 1)
                {
                    rets.AddRange(AddCandidate(call, input, [i, j], [laneSize, laneSize]));
                }
            }
        }

        return rets;
    }
}

public sealed class VectorizeExpand : VectorizeRule
{
    public VectorizeExpand(int rank, int lane)
        : base(rank, lane)
    {
    }

    public override Pattern Pattern { get; } = IsExpand(
      "target",
      "call",
      _ => true,
      IsWildcard("input") with { TypePattern = !IsVector() },
      IsRankedShape("shape"));

    public static List<Expr> AddCandidate(Call call, Expr input, Shape shape, int[] vectorizedAxes, int[] lanes)
    {
        var rets = new List<Expr>();
        var op = (IR.Tensors.Expand)call.Target;
        var inShape = input.CheckedShape;
        if (vectorizedAxes.Length == 0)
        {
            return rets;
        }

        if (vectorizedAxes.Any(a => inShape[a] is { IsFixed: true, FixedValue: 1 }))
        {
            return rets;
        }

        var vectorizedInput = IR.F.Tensors.Pack(VectorizeUtility.PadForVectorize(input, inShape, vectorizedAxes, lanes, 0f, out var padsInput), lanes, vectorizedAxes);

        // only support shape >= input.shape
        var vectorizedNewShape = shape.ToArray();
        foreach (var (lane, axis) in lanes.Zip(vectorizedAxes))
        {
            vectorizedNewShape[axis] = Dimension.CeilDiv(vectorizedNewShape[axis], lane);
        }

        var expand = IR.F.Tensors.Expand(vectorizedInput, vectorizedNewShape);
        var post = VectorizeUtility.SliceForVectorize(IR.F.Tensors.Unpack(expand, lanes, vectorizedAxes), call.CheckedShape, padsInput!);
        if (post.CheckedType is not InvalidType)
        {
            rets.Add(post);
        }

        return rets;
    }

    public override List<Expr> GetReplaceCandidates(IMatchResult result, RunPassContext context)
    {
        var rets = new List<Expr>();
        var call = (Call)result["call"];
        var input = (Expr)result["input"];
        var shape = (RankedShape)result["shape"];
        var laneSize = Lane / input.CheckedDataType.SizeInBytes;

        for (int i = 0; i < input.CheckedShape.Rank; i++)
        {
            rets.AddRange(AddCandidate(call, input, shape, [i], [laneSize]));
        }

        return rets;
    }
}

public sealed class VectorizeWhere : VectorizeRule
{
    public VectorizeWhere(MaskVectorStyle maskVectorStyle, int rank, int lane)
        : base(rank, lane)
    {
        MaskVectorStyle = maskVectorStyle;
    }

    public MaskVectorStyle MaskVectorStyle { get; }

    public override Pattern Pattern { get; } = IsWhere(
        "target",
        "call",
        _ => true,
        IsWildcard("condition") with { TypePattern = !IsMaskVector() },
        IsWildcard("lhs") with { TypePattern = !IsVector() },
        IsWildcard("rhs") with { TypePattern = !IsVector() });

    public static List<Expr> AddCandidate(Expr condition, Expr lhs, Expr rhs, Expr candidate, int[] conditionVectorizedAxes, int[] lhsVectorizedAxes, int[] rhsVectorizedAxes, int[] conditionLanes, int[] lhsLanes, int[] rhsLanes, MaskVectorStyle maskVectorStyle)
    {
        var rets = new List<Expr>();
        var conditionShape = condition.CheckedShape;
        var lhsShape = lhs.CheckedShape;
        var rhsShape = rhs.CheckedShape;
        var outShape = candidate.CheckedShape;
        if (conditionVectorizedAxes.Length == 0 && lhsVectorizedAxes.Length == 0 && rhsVectorizedAxes.Length == 0)
        {
            return rets;
        }

        var alignedConditionVectorizedAxes = conditionVectorizedAxes.Select(a => a + outShape.Rank - conditionShape.Rank).ToArray();
        var alignedLhsVectorizedAxes = lhsVectorizedAxes.Select(a => a + outShape.Rank - lhsShape.Rank).ToArray();
        var alignedRhsVectorizedAxes = rhsVectorizedAxes.Select(a => a + outShape.Rank - rhsShape.Rank).ToArray();
        var union = alignedConditionVectorizedAxes.Union(alignedLhsVectorizedAxes).Union(alignedRhsVectorizedAxes).ToArray();
        if (union.Length > conditionVectorizedAxes.Length && union.Length > lhsVectorizedAxes.Length && union.Length > rhsVectorizedAxes.Length)
        {
            return rets;
        }

        if (conditionVectorizedAxes.Any(a => conditionShape[a] is { IsFixed: true, FixedValue: var d } && d == 1)
            || lhsVectorizedAxes.Any(a => lhsShape[a] is { IsFixed: true, FixedValue: var d } && d == 1)
            || rhsVectorizedAxes.Any(a => rhsShape[a] is { IsFixed: true, FixedValue: var d } && d == 1))
        {
            return rets;
        }

        var alignedCondShape = Enumerable.Repeat(new DimConst(1), outShape.Rank - conditionShape.Rank).Concat(conditionShape).ToArray();
        var alignedLhsShape = Enumerable.Repeat(new DimConst(1), outShape.Rank - lhsShape.Rank).Concat(lhsShape).ToArray();
        var alignedRhsShape = Enumerable.Repeat(new DimConst(1), outShape.Rank - rhsShape.Rank).Concat(rhsShape).ToArray();
        if (alignedConditionVectorizedAxes.Any(a => (alignedLhsShape[a] == alignedCondShape[a] && !alignedLhsVectorizedAxes.Contains(a)) || (alignedLhsShape[a] != alignedCondShape[a] && alignedLhsShape[a] != new DimConst(1)))
        || alignedConditionVectorizedAxes.Any(a => (alignedRhsShape[a] == alignedCondShape[a] && !alignedRhsVectorizedAxes.Contains(a)) || (alignedRhsShape[a] != alignedCondShape[a] && alignedRhsShape[a] != new DimConst(1)))
        || alignedLhsVectorizedAxes.Any(a => (alignedCondShape[a] == alignedLhsShape[a] && !alignedConditionVectorizedAxes.Contains(a)) || (alignedCondShape[a] != alignedLhsShape[a] && alignedCondShape[a] != new DimConst(1)))
        || alignedLhsVectorizedAxes.Any(a => (alignedRhsShape[a] == alignedLhsShape[a] && !alignedRhsVectorizedAxes.Contains(a)) || (alignedRhsShape[a] != alignedLhsShape[a] && alignedRhsShape[a] != new DimConst(1)))
        || alignedRhsVectorizedAxes.Any(a => (alignedCondShape[a] == alignedRhsShape[a] && !alignedConditionVectorizedAxes.Contains(a)) || (alignedCondShape[a] != alignedRhsShape[a] && alignedCondShape[a] != new DimConst(1)))
        || alignedRhsVectorizedAxes.Any(a => (alignedLhsShape[a] == alignedRhsShape[a] && !alignedLhsVectorizedAxes.Contains(a)) || (alignedLhsShape[a] != alignedRhsShape[a] && alignedLhsShape[a] != new DimConst(1))))
        {
            return rets;
        }

        var maskElementBits = lhs.CheckedDataType.SizeInBytes * 8;
        var paddedCondition = VectorizeUtility.PadForVectorize(condition, conditionShape, conditionVectorizedAxes, conditionLanes, 0f, out var conditionPadNums);
        var vectorizedCondition = conditionLanes.Length == 0 ? condition : IR.F.Tensors.VectorizeMask(paddedCondition, maskVectorStyle, maskElementBits, conditionLanes.Single(), conditionVectorizedAxes.Single());
        var vectorizedLhs = IR.F.Tensors.Pack(VectorizeUtility.PadForVectorize(lhs, lhsShape, lhsVectorizedAxes, lhsLanes, 0f, out var lhsPadNums), lhsLanes, lhsVectorizedAxes);
        var vectorizedRhs = IR.F.Tensors.Pack(VectorizeUtility.PadForVectorize(rhs, rhsShape, rhsVectorizedAxes, rhsLanes, 0f, out var rhsPadNums), rhsLanes, rhsVectorizedAxes);

        var compare = IR.F.Tensors.Where(vectorizedCondition, vectorizedLhs, vectorizedRhs);
        var allLanes = new[] { conditionLanes, lhsLanes, rhsLanes };
        var maxIndex = Enumerable.Range(0, allLanes.Length).OrderByDescending(i => allLanes[i].Length).First();
        var outLanes = allLanes[maxIndex];
        var outVectorizeAxes = new[] { alignedConditionVectorizedAxes, alignedLhsVectorizedAxes, alignedRhsVectorizedAxes }.ElementAt(maxIndex);
        var outPadNums = new[] { conditionPadNums, lhsPadNums, rhsPadNums }.ElementAt(maxIndex);
        var post = VectorizeUtility.SliceForVectorize(IR.F.Tensors.Unpack(compare, outLanes, outVectorizeAxes), candidate.CheckedShape, outPadNums);
        if (post.CheckedType is not InvalidType)
        {
            rets.Add(post);
        }

        return rets;
    }

    public override List<Expr> GetReplaceCandidates(IMatchResult result, RunPassContext context)
    {
        var rets = new List<Expr>();
        var condition = (Expr)result["condition"];
        var lhs = (Expr)result["lhs"];
        var rhs = (Expr)result["rhs"];
        var candidate = (Expr)result[Pattern];
        var conditionShape = condition.CheckedShape;
        var lhsShape = lhs.CheckedShape;
        var rhsShape = rhs.CheckedShape;
        var lhsLaneSize = Lane / lhs.CheckedDataType.SizeInBytes;
        var rhsLaneSize = Lane / rhs.CheckedDataType.SizeInBytes;

        // not supoort different lane size.
        var conditionLaneSize = lhsLaneSize;

        foreach (var arr in new[] { GenerateVectorizeAxes(conditionShape), GenerateVectorizeAxes(lhsShape), GenerateVectorizeAxes(rhsShape) }.CartesianProduct())
        {
            var conditionVectorizedAxes = arr.First();
            var lhsVectorizedAxes = arr.Skip(1).First();
            var rhsVectorizedAxes = arr.Skip(2).First();
            if (conditionVectorizedAxes.Length <= Rank && lhsVectorizedAxes.Length <= Rank && rhsVectorizedAxes.Length <= Rank
                && conditionVectorizedAxes.Length == 1 /* only support one axis for condition */)
            {
                rets.AddRange(AddCandidate(condition, lhs, rhs, candidate, conditionVectorizedAxes, lhsVectorizedAxes, rhsVectorizedAxes, Enumerable.Repeat(conditionLaneSize, conditionVectorizedAxes.Length).ToArray(), Enumerable.Repeat(lhsLaneSize, lhsVectorizedAxes.Length).ToArray(), Enumerable.Repeat(rhsLaneSize, rhsVectorizedAxes.Length).ToArray(), MaskVectorStyle));
            }
        }

        return rets;
    }
}

public sealed class VectorizeGather : VectorizeRule
{
    public VectorizeGather(int rank, int lane)
        : base(rank, lane)
    {
    }

    public override Pattern Pattern { get; } = IsGather(
        "target",
        "call",
        _ => true,
        IsWildcard("input") with { TypePattern = !IsVector() },
        IsWildcard("index"));

    public static List<Expr> AddCandidate(Call call, Expr input, Expr index, int[] vectorizedAxes, int[] lanes)
    {
        var rets = new List<Expr>();
        var op = (IR.Tensors.Gather)call.Target;
        var axis = op.Axis;
        var inShape = input.CheckedShape;
        if (vectorizedAxes.Length == 0 || vectorizedAxes.Contains(axis))
        {
            return rets;
        }

        var vectorizedInput = IR.F.Tensors.Pack(VectorizeUtility.PadForVectorize(input, inShape, vectorizedAxes, lanes, 0f, out var padsInput), lanes, vectorizedAxes);

        var gather = IR.F.Tensors.Gather(vectorizedInput, axis, index);
        var devectorizeAxes = vectorizedAxes.Select(a => a < axis ? a : a + index.CheckedShape.Rank - 1).ToArray();
        var post = VectorizeUtility.SliceForVectorize(IR.F.Tensors.Unpack(gather, lanes, devectorizeAxes), call.CheckedShape, padsInput!);
        if (post.CheckedType is not InvalidType)
        {
            rets.Add(post);
        }

        return rets;
    }

    public override List<Expr> GetReplaceCandidates(IMatchResult result, RunPassContext context)
    {
        var rets = new List<Expr>();
        var call = (Call)result["call"];
        var input = (Expr)result["input"];
        var index = (Expr)result["index"];
        var laneSize = Lane / input.CheckedDataType.SizeInBytes;

        for (int i = 0; i < input.CheckedShape.Rank; i++)
        {
            rets.AddRange(AddCandidate(call, input, index, [i], [laneSize]));
        }

        return rets;
    }
}

public sealed class VectorizeScatterND : VectorizeRule
{
    public VectorizeScatterND(int rank, int lane)
        : base(rank, lane)
    {
    }

    public override Pattern Pattern { get; } = IsScatterND(
        "target",
        "call",
        _ => true,
        IsWildcard("input") with { TypePattern = !IsVector() },
        IsTensorConst("indices"),
        IsWildcard("updates") with { TypePattern = !IsVector() });

    public static List<Expr> AddCandidate(Call call, Expr input, TensorConst indices, Expr updates, int[] vectorizedAxes, int[] lanes)
    {
        var rets = new List<Expr>();
        var op = (IR.Tensors.ScatterND)call.Target;
        var inShape = input.CheckedShape;
        var indicesShape = indices.CheckedShape.ToValueArray();
        if (vectorizedAxes.Length == 0 || Enumerable.Range(0, (int)indicesShape[^1]).Intersect(vectorizedAxes).Any())
        {
            return rets;
        }

        var vectorizedInput = IR.F.Tensors.Pack(VectorizeUtility.PadForVectorize(input, inShape, vectorizedAxes, lanes, 0f, out var padsInput), lanes, vectorizedAxes);
        var updatesVectorizedAxes = vectorizedAxes.Select(a => a - (inShape.Rank - updates.CheckedShape.Rank)).ToArray();
        var vectorizedUpdates = IR.F.Tensors.Pack(VectorizeUtility.PadForVectorize(updates, updates.CheckedShape, updatesVectorizedAxes, lanes, 0f, out var padsUpdates), lanes, updatesVectorizedAxes);

        var scatter_nd = IR.F.Tensors.ScatterND(vectorizedInput, indices, vectorizedUpdates);
        var post = VectorizeUtility.SliceForVectorize(IR.F.Tensors.Unpack(scatter_nd, lanes, vectorizedAxes), inShape, padsInput!);
        if (post.CheckedType is not InvalidType)
        {
            rets.Add(post);
        }

        return rets;
    }

    public override List<Expr> GetReplaceCandidates(IMatchResult result, RunPassContext context)
    {
        var rets = new List<Expr>();
        var call = (Call)result["call"];
        var input = (Expr)result["input"];
        var indices = (TensorConst)result["indices"];
        var updates = (Expr)result["updates"];
        var laneSize = Lane / input.CheckedDataType.SizeInBytes;

        for (int i = 0; i < input.CheckedShape.Rank; i++)
        {
            rets.AddRange(AddCandidate(call, input, indices, updates, [i], [laneSize]));
        }

        return rets;
    }
}

public class VectorizeSoftmax : VectorizeRule
{
    public VectorizeSoftmax(int rank, int lane)
        : base(rank, lane)
    {
    }

    public override Pattern Pattern { get; } = IsSoftmax(
      "target",
      IsWildcard("input") with { TypePattern = IsFloat() & !IsVector() },
      IsWildcard("axis") with { TypePattern = IsIntegralScalar() });

    public static List<Expr> AddCandidate(Expr input, int axis, int[] vectorizedAxes, int[] lanes)
    {
        var rets = new List<Expr>();
        var inShape = input.CheckedShape;
        var vectorized = IR.F.Tensors.Pack(VectorizeUtility.PadForVectorize(input, inShape, vectorizedAxes, lanes, float.NegativeInfinity, out var pads), lanes, vectorizedAxes);
        var softmax = IR.F.NTT.VectorizedSoftmax(vectorized, axis, vectorizedAxes);
        if (softmax.CheckedType is not InvalidType)
        {
            var post = VectorizeUtility.SliceForVectorize(IR.F.Tensors.Unpack(softmax, lanes, vectorizedAxes), inShape, pads);
            rets.Add(post);
        }

        return rets;
    }

    public override List<Expr> GetReplaceCandidates(IMatchResult result, RunPassContext context)
    {
        var rets = new List<Expr>();
        var input = (Expr)result["input"];
        var axis = (int)((DimConst)result["axis"]).FixedValue;
        var inShape = input.CheckedShape;
        var laneSize = Lane / input.CheckedDataType.SizeInBytes;

        for (int i = 0; i < inShape.Rank; i++)
        {
            rets.AddRange(AddCandidate(input, axis, new[] { i }, new[] { laneSize }));
            for (int j = i + 1; j < inShape.Rank; j++)
            {
                if (Rank > 1)
                {
                    rets.AddRange(AddCandidate(input, axis, new[] { i, j }, new[] { laneSize, laneSize }));
                }
            }
        }

        return rets;
    }
}

public sealed class VectorizeLayerNorm : VectorizeRule
{
    public VectorizeLayerNorm(int rank, int lane)
        : base(rank, lane)
    {
    }

    public override Pattern Pattern { get; } = IsLayerNorm(
      "target",
      _ => true,
      IsWildcard("input") with { TypePattern = IsFloat() & !IsVector() },
      IsWildcard("scale") with { TypePattern = IsFloat() & !IsVector() },
      IsWildcard("bias") with { TypePattern = IsFloat() & !IsVector() });

    public static List<Expr> AddCandidate(LayerNorm op, Expr input, Expr scale, Expr bias, int[] vectorizedAxes, int[] lanes)
    {
        var rets = new List<Expr>();
        var inShape = input.CheckedShape;
        var pshape = scale.CheckedShape;
        var vectorizedInput = IR.F.Tensors.Pack(VectorizeUtility.PadForVectorize(input, inShape, vectorizedAxes, lanes, 0f, out var padsInput), lanes, vectorizedAxes);

        var pAxes = vectorizedAxes.Where(i => i >= op.Axis).Select(i => i - op.Axis).ToArray();
        var vectorizedScale = VectorizeUtility.PadForVectorize(scale, pshape, pAxes, lanes, 0f, out var padsScale);
        if (pAxes.Length > 0)
        {
            vectorizedScale = IR.F.Tensors.Pack(vectorizedScale, Enumerable.Repeat(lanes[0], pAxes.Length).ToArray(), pAxes);
        }

        var vectorizedBias = VectorizeUtility.PadForVectorize(bias, pshape, pAxes, lanes, 0f, out var padsBias);
        if (pAxes.Length > 0)
        {
            vectorizedBias = IR.F.Tensors.Pack(vectorizedBias, Enumerable.Repeat(lanes[0], pAxes.Length).ToArray(), pAxes);
        }

        var layernorm = IR.F.NTT.VectorizedLayerNorm(vectorizedInput, vectorizedScale, vectorizedBias, op.Axis, op.Epsilon, op.UseMean, vectorizedAxes, new RankedShape(padsInput));

        if (layernorm.CheckedType is not InvalidType)
        {
            var post = VectorizeUtility.SliceForVectorize(IR.F.Tensors.Unpack(layernorm, lanes, vectorizedAxes), inShape, padsInput);
            rets.Add(post);
        }

        return rets;
    }

    public override List<Expr> GetReplaceCandidates(IMatchResult result, RunPassContext context)
    {
        var rets = new List<Expr>();
        var op = (IR.NN.LayerNorm)result["target"];
        var input = (Expr)result["input"];
        var scale = (Expr)result["scale"];
        var bias = (Expr)result["bias"];
        var laneSize = Lane / input.CheckedDataType.SizeInBytes;

        for (int i = 0; i < input.CheckedShape.Rank; i++)
        {
            rets.AddRange(AddCandidate(op, input, scale, bias, new[] { i }, new[] { laneSize }));
            for (int j = i + 1; j < input.CheckedShape.Rank; j++)
            {
                if (Rank > 1)
                {
                    rets.AddRange(AddCandidate(op, input, scale, bias, new[] { i, j }, new[] { laneSize, laneSize }));
                }
            }
        }

        return rets;
    }
}

[RuleGenerator]
public sealed partial class FoldVectorizeDevectorize : RewriteRule<Pattern>
{
    public override Pattern Pattern { get; } = PatternMatch.F.Tensors.IsPack("vectorize", "caller", _ => true, PatternMatch.F.Tensors.IsUnpack("devectorize", "callee", _ => true, IsWildcard("input")));

    private Expr? GetReplace(IR.Tensors.Pack vectorize, IR.Tensors.Unpack devectorize, Expr input)
    {
        if (vectorize.Axes.SequenceEqual(devectorize.Axes) && vectorize.Lanes.SequenceEqual(devectorize.Lanes))
        {
            return input;
        }

        return null;
    }
}

[RuleGenerator]
public sealed partial class FoldVectorizeConcatDevectorize : RewriteRule<Pattern>
{
    public override Pattern Pattern { get; } = PatternMatch.F.Tensors.IsPack("vectorize", "caller", _ => true, PatternMatch.F.Tensors.IsConcat("concat", _ => true, IsTuple("tuple", IsVArgsRepeat("fileds", exprs =>
        {
            var patterns = new Pattern[exprs.Length];
            for (int i = 0; i < exprs.Length; i++)
            {
                patterns[i] = PatternMatch.F.Tensors.IsUnpack($"devectorize_{i}", $"callee_{i}", _ => true, IsWildcard($"input_{i}"));
            }

            return patterns;
        }))));

    private Expr? GetReplace(IR.Tensors.Pack vectorize, IR.Tensors.Concat concat, IReadOnlyList<BaseExpr> fileds, IMatchResult result)
    {
        var inputs = new Expr[fileds.Count];
        for (int i = 0; i < fileds.Count; i++)
        {
            var devectorize = (IR.Tensors.Unpack)result[$"devectorize_{i}"];
            if (vectorize.Axes.SequenceEqual(devectorize.Axes) && vectorize.Lanes.SequenceEqual(devectorize.Lanes))
            {
                inputs[i] = (Expr)result[$"input_{i}"];
            }
            else
            {
                return null;
            }
        }

        return IR.F.Tensors.Concat(new IR.Tuple(inputs), concat.Axis);
    }
}

[RuleGenerator]
public sealed partial class TransposeVectorizeMatMulInputs : RewriteRule<Pattern>
{
    public override Pattern Pattern { get; } = PatternMatch.F.NTT.IsVectorizedMatMul("matmul", "caller", m => m.RhsVectorizedAxes.Count == 2 && m.TransposeB == false, IsWildcard("lhs"), PatternMatch.F.Tensors.IsPack("rhsVectorize", "callee", p => p.Axes.Count == 2 && p.Lanes.Count == 2, PatternMatch.F.Tensors.IsTranspose("trans", "rhs", IsWildcard("transInput"), IsFixedShape("perm") /* IsAlt(IsTensorConst("rhs"), PatternMatch.F.Tensors.IsTranspose("trans", "rhs", IsWildcard("transInput"), IsTensorConst("perm")) */)));

    private Expr? GetReplace(IR.NTT.VectorizedMatMul matmul, Expr lhs, IR.Tensors.Pack rhsVectorize, Expr transInput, int[] perm, IMatchResult result)
    {
        // note can't enable transpose const b, because const folding will be very solw.
        var inputsShape = transInput.CheckedShape.ToValueArray();
        var tperm = Enumerable.Range(0, inputsShape.Length).ToArray();
        (tperm[^2], tperm[^1]) = (tperm[^1], tperm[^2]);
        if (tperm.SequenceEqual(perm))
        {
            var nvectorize = IR.F.Tensors.Pack(transInput, [rhsVectorize.Lanes[1], rhsVectorize.Lanes[0]], [rhsVectorize.Axes[1], rhsVectorize.Axes[0]]);
            var newMatmul = new IR.NTT.VectorizedMatMul(matmul.OutputDataType, matmul.LhsVectorizedAxes, new[] { matmul.RhsVectorizedAxes[1], matmul.RhsVectorizedAxes[0] }, false, true, matmul.FusedReduce);
            return new Call(newMatmul, lhs, nvectorize);
        }

        return null;
    }
}

[RuleGenerator]
public sealed partial class FoldNopVectorize : RewriteRule<Pattern>
{
    public override Pattern Pattern { get; } = PatternMatch.F.Tensors.IsPack("vectorize", "call", _ => true, IsWildcard("input"));

    private Expr? GetReplace(IR.Tensors.Pack vectorize, Expr input)
    {
        if (vectorize.Axes.Count == 0)
        {
            return input;
        }

        return null;
    }
}

[RuleGenerator]
public sealed partial class FoldNopDevectorize : RewriteRule<Pattern>
{
    public override Pattern Pattern { get; } = PatternMatch.F.Tensors.IsUnpack("devectorize", "call", _ => true, IsWildcard("input"));

    private Expr? GetReplace(IR.Tensors.Unpack devectorize, Expr input)
    {
        if (devectorize.Axes.Count == 0)
        {
            return input;
        }

        return null;
    }
}
