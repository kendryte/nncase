// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using DryIoc.ImTools;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using Nncase.PatternMatch;
using Nncase.Utilities;
using static Nncase.IR.F.Math;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Neutral;

/// <summary>
/// Fold nop <see cref="IR.Tensors.Reshape"/>.
/// </summary>
[RuleGenerator]
public sealed partial class FoldNopReshape : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } = IsReshape(
        IsWildcard("input") with { TypePattern = HasFixedShape() },
        IsTensorConst("newShape", IsIntegral()));

    private Expr? GetReplace(Expr input, TensorConst newShape)
    {
        var newShapeArray = newShape.Value.ToArray<int>();
        if ((newShapeArray.Count(x => x == -1) == 1 && newShapeArray.Length == input.CheckedShape.Count
             && input.CheckedShape.Zip(newShapeArray).Count(t => t.Second != -1 && t.First.FixedValue == t.Second) == newShapeArray.Length - 1)
            || input.CheckedShape.ToValueArray().SequenceEqual(newShapeArray))
        {
            return input;
        }

        return null;
    }
}

/// <summary>
/// Fold two <see cref="IR.Tensors.Reshape"/>.
/// </summary>
[RuleGenerator]
public sealed partial class FoldTwoReshapes : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } = IsReshape(
        IsReshape(IsWildcard("input"), IsWildcard()), IsWildcard("newShape"));

    private Expr? GetReplace(Expr input, Expr newShape)
    {
        return Reshape(input, newShape);
    }
}

/// <summary>
/// Fold sequeeze reshape(binary(unsequeeze reshape(x), const)).
/// </summary>
[RuleGenerator]
public sealed partial class FoldReshapeBinaryConstReshape : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } =
        IsReshape(IsSwappableBinary("binary", null, b => b.BinaryOp is BinaryOp.Add or BinaryOp.Mul, IsReshape(IsWildcard("input") with { TypePattern = HasFixedShape() }, IsTensorConst("unsqShape")), IsTensorConst("binaryConst")), IsTensorConst("sqShape"));

    private Expr? GetReplace(Expr input, Binary binary, int[] unsqShape, TensorConst binaryConst, int[] sqShape)
    {
        var inShape = input.CheckedShape.ToValueArray();
        if (!(sqShape.SequenceEqual(inShape) && RulesUtility.FindSqueezeAxis(unsqShape, sqShape) is int axis && axis != -1 && (
            (binaryConst.Value.Shape.Rank == unsqShape.Length && binaryConst.Value.Shape[axis].Value == 1) || (Evaluator.TypeInference.BroadcastType((TensorType)input.CheckedType, (TensorType)binaryConst.CheckedType) is TensorType outType && outType.Shape.ToValueArray().SequenceEqual(inShape)))))
        {
            return null;
        }

        return IR.F.Math.Binary(binary.BinaryOp, input, (binaryConst.Value.Shape.Rank == unsqShape.Length && binaryConst.Value.Shape[axis].Value == 1) ? IR.F.Tensors.Squeeze(binaryConst, new[] { axis }) : binaryConst);
    }
}

/// <summary>
/// Fold nop <see cref="IR.Tensors.Reshape"/>.
/// </summary>
[RuleGenerator]
public sealed partial class ReshapeToTranspose : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } = IsReshape(
        "reshape",
        "call",
        _ => true,
        IsWildcard("input") with { TypePattern = HasFixedShape() },
        IsTensorConst("newShape", IsIntegral()));

    private Expr? GetReplace(Expr input, Call call)
    {
        if (input.CheckedShape.Rank <= 1)
        {
            return null;
        }

        var newShape = call.CheckedShape.ToValueArray();
        var inShape = input.CheckedShape.ToValueArray();
        var sigNewShape = newShape.Where(x => x != 1).ToArray();
        var sigInShape = inShape.Where(x => x != 1).ToArray();
        if (newShape.Length == inShape.Length && sigInShape.SequenceEqual(sigNewShape))
        {
            var inShapeList = inShape.Zip(Enumerable.Range(0, inShape.Length)).ToList();
            var perm = new List<int>();
            for (var o = 0; o < newShape.Length; o++)
            {
                var inShapeZip = inShapeList.FindFirst((i) => i.First == newShape[o]);
                perm.Add(inShapeZip.Second);
                inShapeList.Remove(inShapeZip);
            }

            return Transpose(input, perm.ToArray()).InheritMetaData(call);
        }

        return null;
    }
}
