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
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Neutral;

/// <summary>
/// Fold <see cref="IR.Tensors.Cast"/> by const scalar.
/// </summary>
[RuleGenerator]
public sealed partial class FoldCastPostOps : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } =
        IsSwappableBinary(
            "binary",
            null,
            b => b.BinaryOp is BinaryOp.Add or BinaryOp.Mul,
            IsCast(
                "cast",
                null,
                _ => true,
                IsWildcard("input"),
                IsWildcard("postOp") with { TypePattern = IsNoneType() }),
            IsTensorConst("constScalar") with { TypePattern = IsScalar() | HasShape(new Dimension[] { new DimConst(1) }) });

    private Expr? GetReplace(IR.Tensors.Cast cast, Expr input, IR.Math.Binary binary, TensorConst constScalar)
    {
        var @var = new Var(AnyType.Default);
        var scalar = Tensor.FromBytes(constScalar.Value.ElementType, constScalar.Value.BytesBuffer.ToArray(), []);
        var postOp = new Fusion(CompileSessionScope.Current!.Target.Name, IR.F.Math.Binary(binary.BinaryOp, @var, scalar), @var);
        return IR.F.Tensors.Cast(input, cast.NewType, cast.CastMode, cast.VectorizeAxes, postOp);
    }
}

/// <summary>
/// Fold <see cref="IR.Math.Binary"/> by const scalar.
/// </summary>
[RuleGenerator]
public sealed partial class FoldBinaryPostOps : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } =
        IsSwappableBinary(
            "binary2",
            null,
            b => b.BinaryOp is BinaryOp.Add or BinaryOp.Mul,
            IsBinary(
                "binary1",
                null,
                _ => true,
                IsWildcard("lhs"),
                IsWildcard("rhs"),
                IsWildcard("postOp") with { TypePattern = IsNoneType() }),
            IsTensorConst("constScalar") with { TypePattern = IsScalar() | HasShape(new Dimension[] { new DimConst(1) }) });

    private Expr? GetReplace(IR.Math.Binary binary1, Expr lhs, Expr rhs, IR.Math.Binary binary2, TensorConst constScalar)
    {
        var @var = new Var(AnyType.Default);
        var scalar = Tensor.FromBytes(constScalar.Value.ElementType, constScalar.Value.BytesBuffer.ToArray(), []);
        var postOp = new Fusion(CompileSessionScope.Current!.Target.Name, IR.F.Math.Binary(binary2.BinaryOp, @var, scalar), @var);
        return IR.F.Math.Binary(binary1.BinaryOp, lhs, rhs, postOp);
    }
}
