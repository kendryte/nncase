// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.IR;
using Nncase.PatternMatch;
using static Nncase.IR.F.Math;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Neutral;

/// <summary>
/// Fold nop <see cref="IR.Math.Binary"/>.
/// </summary>
[RuleGenerator]
public sealed partial class ExpandToBinary : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } = PatternMatch.F.Tensors.IsExpand(
        "expand",
        "call",
        IsWildcard("input"),
        IsTensorConst("shape"));

    private Expr? GetReplace(Expr input, TensorConst shape)
    {
        return input.CheckedDataType switch
        {
            var x when x == DataTypes.UInt8 => Binary(BinaryOp.Mul, input, Tensor.FromScalar((byte)1, shape.Value.ToArray<int>())),
            var x when x == DataTypes.UInt16 => Binary(BinaryOp.Mul, input, Tensor.FromScalar((ushort)1, shape.Value.ToArray<int>())),
            var x when x == DataTypes.UInt32 => Binary(BinaryOp.Mul, input, Tensor.FromScalar(1U, shape.Value.ToArray<int>())),
            var x when x == DataTypes.UInt64 => Binary(BinaryOp.Mul, input, Tensor.FromScalar(1UL, shape.Value.ToArray<int>())),
            var x when x == DataTypes.Int8 => Binary(BinaryOp.Mul, input, Tensor.FromScalar((sbyte)1, shape.Value.ToArray<int>())),
            var x when x == DataTypes.Int16 => Binary(BinaryOp.Mul, input, Tensor.FromScalar((short)1, shape.Value.ToArray<int>())),
            var x when x == DataTypes.Int32 => Binary(BinaryOp.Mul, input, Tensor.FromScalar(1, shape.Value.ToArray<int>())),
            var x when x == DataTypes.Int64 => Binary(BinaryOp.Mul, input, Tensor.FromScalar(1L, shape.Value.ToArray<int>())),
            var x when x == DataTypes.Float16 => Binary(BinaryOp.Mul, input, Tensor.FromScalar((Half)1, shape.Value.ToArray<int>())),
            var x when x == DataTypes.Float32 => Binary(BinaryOp.Mul, input, Tensor.FromScalar(1F, shape.Value.ToArray<int>())),
            var x when x == DataTypes.Float64 => Binary(BinaryOp.Mul, input, Tensor.FromScalar(1D, shape.Value.ToArray<int>())),
            _ => null,
        };
    }
}
