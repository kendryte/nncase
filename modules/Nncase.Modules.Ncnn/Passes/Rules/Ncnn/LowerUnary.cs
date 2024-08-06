// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
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
public partial class LowerUnary : RewriteRule<Pattern>
{
    /// <inheritdoc/>
    public override Pattern Pattern { get; } = IsUnary(
      target_name: "unary",
      _ => true,
      IsWildcard("input") with { TypePattern = IsFloat() & HasFixedShape() });

    private static UnaryOperationType? MapUnaryOp(UnaryOp unaryOp) =>
        unaryOp switch
        {
            UnaryOp.Abs => UnaryOperationType.ABS,
            UnaryOp.Neg => UnaryOperationType.NEG,
            UnaryOp.Floor => UnaryOperationType.FLOOR,
            UnaryOp.Ceil => UnaryOperationType.CEIL,
            UnaryOp.Square => UnaryOperationType.SQUARE,
            UnaryOp.Sqrt => UnaryOperationType.SQRT,
            UnaryOp.Rsqrt => UnaryOperationType.RSQRT,
            UnaryOp.Exp => UnaryOperationType.EXP,
            UnaryOp.Log => UnaryOperationType.LOG,
            UnaryOp.Sin => UnaryOperationType.SIN,
            UnaryOp.Cos => UnaryOperationType.COS,
            UnaryOp.Asin => UnaryOperationType.ASIN,
            UnaryOp.Acos => UnaryOperationType.ACOS,
            UnaryOp.Tanh => UnaryOperationType.TANH,
            UnaryOp.Round => UnaryOperationType.ROUND,
            _ => null,

            // unsupported unary ops
            // UnaryOp.TAN => UnaryOperationType.ABS,
            // UnaryOp.Atan => UnaryOperationType.ATAN,
            // UnaryOp.Reciprocal => UnaryOperationType.RECIPROCAL,
            // UnaryOp.Log10 => UnaryOperationType.LOG10,
            // UnaryOp.Trunc => UnaryOperationType.TRUNC,
        };

    // squeeze unary shape to 3D
    private List<int> GetFixedShape(List<int> oldShape)
    {
        var newShape = new List<int>();

        newShape.AddRange(oldShape.GetRange(oldShape.Count - 3, 3));

        for (int i = 0; i < oldShape.Count - 3; i++)
        {
            newShape[0] *= oldShape[i];
        }

        return newShape;
    }

    private Expr? GetReplace(Unary unary, Expr input)
    {
        if (MapUnaryOp(unary.UnaryOp) is UnaryOperationType op)
        {
            var newInput = new Var(input.CheckedType);
            if (input.CheckedShape.Rank > 3)
            {
                var newShape = GetFixedShape(input.CheckedShape.ToValueList());

                var inRes = Reshape(input, newShape.ToArray());
                var inResO = new Var(inRes.CheckedType);

                var ncnnUnaryCall = new Call(new Fusion("ncnn", NcnnUnary(inResO, op), new[] { inResO }), inRes);

                var outRes = Reshape(ncnnUnaryCall, input.CheckedShape.ToValueList().ToArray());
                return outRes;
            }
            else
            {
                return new Call(new Fusion("ncnn", NcnnUnary(newInput, op), new[] { newInput }), input);
            }
        }

        return null;
    }
}
