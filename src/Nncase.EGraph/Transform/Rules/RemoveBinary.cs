// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using Nncase.IR;
using Nncase.Pattern;
using Nncase.Pattern.Math;
using static Nncase.Pattern.Utility;
using static Nncase.IR.F.Tensors;

namespace Nncase.Transform.Rule
{
    public sealed class RemoveNoSenceBinary : PatternRule
    {
        private BinaryWrapper binary;

        public RemoveNoSenceBinary()
        {
            Pattern = binary = IsBinary(x => x is (BinaryOp.Add or BinaryOp.Sub or BinaryOp.Mul or BinaryOp.Div), IsWildCard(), IsWildCard());
        }

        private bool CheckValue(TensorConst con, float value) =>
          con.ValueType.IsScalar ?
            con.Value.ToScalar<float>() == value :
            con.Value.Cast<float>().All(v => v == value);

        public override Expr? GetRePlace(IMatchResult result)
        {
            binary.Bind(result);
            var binaryOp = binary.BinaryOp;
            var newexpr = (binaryOp, binary.Lhs(), binary.Rhs()) switch
            {
                (BinaryOp.Add, TensorConst lhs, Expr rhs) => CheckValue(lhs, 0) ? rhs : null,
                (BinaryOp.Add, Expr lhs, TensorConst rhs) => CheckValue(rhs, 0) ? lhs : null,

                (BinaryOp.Sub, Expr lhs, TensorConst rhs) => CheckValue(rhs, 0) ? lhs : null,

                (BinaryOp.Mul, TensorConst lhs, Expr rhs) => CheckValue(lhs, 1) ? rhs : null,
                (BinaryOp.Mul, Expr lhs, TensorConst rhs) => CheckValue(rhs, 1) ? lhs : null,

                (BinaryOp.Div, Expr lhs, TensorConst rhs) => CheckValue(rhs, 1) ? lhs : null,
                (_, _, _) => null,
            };
            if (newexpr is not null)
            {
                var out_shape = result[binary].CheckedShape;
                if (out_shape != newexpr.CheckedShape)
                {
                    return Broadcast(newexpr, out_shape);
                }
            }

            return newexpr;
        }
    }
}