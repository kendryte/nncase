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
using Nncase.IR.Math;
using Nncase.IR.Ncnn;
using Nncase.PatternMatch;

using static Nncase.IR.F.Ncnn;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Ncnn;

[RuleGenerator]
public partial class LowerBinary : RewriteRule<Pattern>
{
    /// <inheritdoc/>
    public override Pattern Pattern { get; } = IsBinary(
      target_name: "binary",
      _ => true,
      IsWildcard("inputA") with { TypePattern = IsFloat() & HasRank(x => x <= 3) },
      IsWildcard("inputB") with { TypePattern = IsFloat() & HasRank(x => x <= 3) });

    private static BinaryOperationType? MapBinaryOp(BinaryOp binaryOp) =>
        binaryOp switch
        {
            BinaryOp.Add => BinaryOperationType.ADD,
            BinaryOp.Sub => BinaryOperationType.SUB,
            BinaryOp.Mul => BinaryOperationType.MUL,
            BinaryOp.Div => BinaryOperationType.DIV,
            BinaryOp.Max => BinaryOperationType.MAX,
            BinaryOp.Min => BinaryOperationType.MIN,
            BinaryOp.Pow => BinaryOperationType.POW,
            _ => null,

            // unsupported Binary ops
            // BinaryOp.Mod =>
            // BitwiseAnd
            // BitwiseOr
            // BitwiseXor
            // LogicalAnd
            // LogicalOr
            // LogicalXor
            // LeftShift
            // RightShift
            // => BinaryOperationType.RSUB,
            // => BinaryOperationType.RDIV,
            // => BinaryOperationType.RPOW,
            // => BinaryOperationType.ATAN2,
            // => BinaryOperationType.RATAN2,
        };

    private int[] FixShape(int[] shape, int r)
    {
        if (shape.Length == 1)
        {
            return shape;
        }

        var newShape = shape.ToList();
        for (int i = r - shape.Length; i > 0; i--)
        {
            newShape.Insert(0, 1);
        }

        return newShape.ToArray();
    }

    private Expr? GetReplace(Binary binary, Expr inputA, Expr inputB)
    {
        if (MapBinaryOp(binary.BinaryOp) is BinaryOperationType op)
        {

            var r = Math.Max(inputA.CheckedShape.Rank, inputB.CheckedShape.Rank);
            if (inputA is Const)
            {
                var constA = ((TensorConst)inputA).Value;
                var constShape = FixShape(inputA.CheckedShape.ToValueArray(), r);
                var newB = Reshape(inputB, FixShape(inputB.CheckedShape.ToValueArray(), r));
                var newInputB = new Var(newB.CheckedType);
                return new Call(new Fusion("ncnn", NcnnBinary(new Expr[] { newInputB }, op, 1, constA.ToArray<float>(), constShape), new[] { newInputB }), newB);
            }
            else if (inputB is Const)
            {
                var newA = Reshape(inputA, FixShape(inputA.CheckedShape.ToValueArray(), r));
                var newInputA = new Var(newA.CheckedType);
                var constB = ((TensorConst)inputB).Value;
                var constShape = FixShape(inputB.CheckedShape.ToValueArray(), r);
                return new Call(new Fusion("ncnn", NcnnBinary(new Expr[] { newInputA }, op, 2, constB.ToArray<float>(), constShape), newInputA), newA);
            }
            else
            {
                var newInputA = new Var(inputA.CheckedType);
                var newInputB = new Var(inputB.CheckedType);
                return new Call(new Fusion("ncnn", NcnnBinary(new Expr[] { newInputA, newInputB }, op, 0, null, null), new[] { newInputA, newInputB }), inputA, inputB);
            }
        }

        return null;
    }
}
