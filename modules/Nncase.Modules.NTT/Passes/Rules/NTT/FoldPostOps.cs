// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using Nncase.PatternMatch;
using static Nncase.IR.F.Math;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.F.NTT;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.NTT;

[RuleGenerator]
public sealed partial class FoldCastPostOps : IRewriteRule
{
    public FoldCastPostOps()
    {
        var cast = IsVectorizedCast(
                "cast",
                "callee",
                _ => true,
                IsWildcard("input"),
                IsWildcard("postOps"));

        var scalar = IsTensorConst("constScalar") with { TypePattern = IsScalar() | HasShape(new Dimension[] { new DimConst(1) }) };

        Pattern = IsAlt(
            IsBinary("binary", "caller", b => b.BinaryOp is BinaryOp.Add or BinaryOp.Mul, cast, scalar),
            IsBinary("binary", "caller", b => b.BinaryOp is BinaryOp.Add or BinaryOp.Mul, scalar, cast));
    }

    /// <inheritdoc/>
    public IPattern Pattern { get; }

    private Expr? GetReplace(IR.NTT.VectorizedCast cast, Expr input, Expr postOps, IR.Math.Binary binary, TensorConst constScalar)
    {
        var scalar = Tensor.FromBytes(constScalar.Value.ElementType, constScalar.Value.BytesBuffer.ToArray(), []);
        Expr newPostOp = None.Default;
        if (postOps is None)
        {
            var @var = new Var(AnyType.Default);
            newPostOp = new Fusion(CompileSessionScope.Current!.Target.Name, IR.F.Math.Binary(binary.BinaryOp, @var, scalar), @var);
        }
        else if (postOps is Fusion fusion)
        {
            newPostOp = fusion.With(body: IR.F.Math.Binary(binary.BinaryOp, (Expr)fusion.Body, scalar));
        }

        return IR.F.NTT.VectorizedCast(input, cast.NewType, cast.CastMode, cast.VectorizeAxes, newPostOp);
    }
}

[RuleGenerator]
public sealed partial class FoldBinaryPostOps : IRewriteRule
{
    public FoldBinaryPostOps()
    {
        var binary1 = IsVectorizedBinary("binary1", "callee", _ => true, IsWildcard("lhs"), IsWildcard("rhs"), IsWildcard("postOps"));
        var scalar = IsTensorConst("constScalar") with { TypePattern = IsScalar() | HasShape(new Dimension[] { new DimConst(1) }) };

        Pattern = IsAlt(
            IsBinary("binary2", "caller", b => b.BinaryOp is BinaryOp.Add or BinaryOp.Mul, binary1, scalar),
            IsBinary("binary2", "caller", b => b.BinaryOp is BinaryOp.Add or BinaryOp.Mul, scalar, binary1));
    }

    /// <inheritdoc/>
    public IPattern Pattern { get; }

    private Expr? GetReplace(IR.NTT.VectorizedBinary binary1, Expr lhs, Expr rhs, Expr postOps, IR.Math.Binary binary2, TensorConst constScalar)
    {
        var scalar = Tensor.FromBytes(constScalar.Value.ElementType, constScalar.Value.BytesBuffer.ToArray(), []);
        Expr newPostOp = None.Default;
        if (postOps is None)
        {
            var @var = new Var(AnyType.Default);
            newPostOp = new Fusion(CompileSessionScope.Current!.Target.Name, IR.F.Math.Binary(binary2.BinaryOp, @var, scalar), @var);
        }
        else if (postOps is Fusion fusion)
        {
            newPostOp = fusion.With(body: IR.F.Math.Binary(binary2.BinaryOp, (Expr)fusion.Body, scalar));
        }

        return IR.F.NTT.VectorizedBinary(lhs, rhs, newPostOp, binary1.BinaryOp, binary1.LhsVectorizedAxes, binary1.LhsPadedNums, binary1.RhsVectorizedAxes, binary1.RhsPadedNums);
    }
}
