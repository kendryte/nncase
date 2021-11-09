using System;
using System.Linq;
using Nncase.IR;
using Nncase.Transform.Pattern;
using Nncase.Transform.Pattern.Math;
using static Nncase.Transform.Pattern.Utility;
using static Nncase.IR.F.Tensors;

namespace Nncase.Transform.Rule
{
    public sealed class Reassociate : EGraphRule
    {
        private WildCardPattern wx = "x", wy = "y", wz = "z";

        public Reassociate()
        {

            Pattern = (wx * wy) * wz;
        }


        public override Expr GetRePlace(EMatchResult result)
        {
            var (x, y, z) = result[wx, wy, wz];
            return x * (y * z);
        }
    }

    public sealed class RemoveNoSenceBinary : EGraphRule
    {
        private BinaryWrapper binary;

        public RemoveNoSenceBinary()
        {
            Pattern = binary = IsBinary(x => x is (BinaryOp.Add or BinaryOp.Sub or BinaryOp.Mul or BinaryOp.Div), IsWildCard(), IsWildCard());
        }

        private bool CheckValue(Const con, float value) =>
          con.ValueType.IsScalar ?
            con.ToScalar<float>() == value :
            con.ToTensor<float>().All(v => v == value);

        public override Expr? GetRePlace(EMatchResult result)
        {
            binary.Bind(result);
            var binaryOp = binary.BinaryOp;
            var newexpr = (binaryOp, binary.Lhs(), binary.Rhs()) switch
            {
                (BinaryOp.Add, Const lhs, Expr rhs) => CheckValue(lhs, 0) ? rhs : null,
                (BinaryOp.Add, Expr lhs, Const rhs) => CheckValue(rhs, 0) ? lhs : null,

                (BinaryOp.Sub, Expr lhs, Const rhs) => CheckValue(rhs, 0) ? lhs : null,

                (BinaryOp.Mul, Const lhs, Expr rhs) => CheckValue(lhs, 1) ? rhs : null,
                (BinaryOp.Mul, Expr lhs, Const rhs) => CheckValue(rhs, 1) ? lhs : null,

                (BinaryOp.Div, Expr lhs, Const rhs) => CheckValue(rhs, 1) ? lhs : null,
                (_, _, _) => null
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