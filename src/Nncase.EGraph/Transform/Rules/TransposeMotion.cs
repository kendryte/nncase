using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.Transform.Pattern;
using static Nncase.Transform.Pattern.F.Tensor;
using static Nncase.Transform.Pattern.Utility;
using static Nncase.IR.F.Tensors;

namespace Nncase.Transform.Rule
{
    public class TransposeBinaryMotion : EGraphRule
    {
        private WildCardPattern wx = "x", wy = "y", wperm = "perm";
        private WildCardPattern wbin = new();

        public override ExprPattern GetPattern()
        {
            return IsBinary(Transpose(wx, wperm), Transpose(wy, wperm));
        }

        public override Expr GetRePlace(EMatchResult result)
        {
            Expr x = result.GetExpr(wx), y = result.GetExpr(wy), perm = result.GetExpr(wperm);
            var bin = (Binary)result.Root.Expr;
            return Transpose(new Call(new Binary(bin.BinaryOp), x, y), perm);
        }
    }

    public class TransposeConstantBinaryMotion : EGraphRule
    {
        protected ID wx = "x", wperm = "perm", wcon = "con";

        protected Expr x;
        protected Const con, perm;
        protected Binary binary;

        protected Shape newShape;
        protected Const newCon;

        public Shape GetNewConstShape(Const oldCon, Const oldPerm)
        {
            var perm = oldPerm.ToTensor<int>();
            var expand_dim = perm.Length - perm[(int)perm.Length - 1] - 1;
            var shape = oldCon.Shape;
            if (shape[0] != 1)
            {
                for (int i = 0; i < expand_dim; i++)
                    shape.Add(1);
            }
            return new Shape(shape);
        }

        public void ParserResult(EMatchResult result)
        {
            x = result.GetExpr(wx);
            con = (Const)result.GetExpr(wcon);
            perm = (Const)result.GetExpr(wperm);
            binary = (Binary)result.GetRoot();
            newShape = GetNewConstShape((Const)con, (Const)perm);
            newCon = con with { ValueType = con.ValueType with { Shape = newShape } };
        }
    }

    public class TransposeConstantBinaryMotionLeft : TransposeConstantBinaryMotion
    {
        public override ExprPattern GetPattern()
        {
            return IsBinary(Transpose(IsWildCard(wx), IsConst(wperm)), IsConst(wcon));
        }

        public override Expr GetRePlace(EMatchResult result)
        {
            ParserResult(result);
            return Transpose(new Call(new Binary(binary.BinaryOp), x, newCon), perm);
        }
    }

    public class TransposeConstantBinaryMotionRight : TransposeConstantBinaryMotion
    {
        public override ExprPattern GetPattern()
        {
            return IsBinary(IsConst(wcon), Transpose(IsWildCard(wx), IsConst(wperm)));
        }

        public override Expr GetRePlace(EMatchResult result)
        {
            ParserResult(result);
            return Transpose(new Call(new Binary(binary.BinaryOp), newCon, x), perm);
        }
    }


    public class TransposeConcatMotion : EGraphRule
    {

        private class Comparer
        {
            private Expr? last_const_ = null;
            public bool Cond(Const con)
            {
                if (last_const_ is null)
                    last_const_ = con;
                return last_const_ == con;
            }
        }
        private Func<Const, bool> permCond = new(new Comparer().Cond);

        ID wcprem = "wcprem", wcaxis = "axis";

        List<ID> wcinputs = new();

        public override ExprPattern GetPattern()
        {
            return Concat(IsTuple(IsVArgsRepeat(n =>
            {
                var ret = new ExprPattern[n];
                for (int i = 0; i < n; i++)
                {
                    var wcin = IsWildCard();
                    ret[i] = Transpose(wcin, IsConst(wcprem, permCond));
                }
                return ret;
            }
            )), IsConst(wcaxis));
        }

        public override Expr GetRePlace(EMatchResult result)
        {
            var newShapes = (from input in wcinputs select GetShape(result.GetExpr(input))).ToArray();
            var oldPerm = result.GetExpr<Const>(wcprem);
            var permt = oldPerm.ToTensor<int>();
            var oldAxis = result.GetExpr<Const>(wcaxis).ToScalar<int>();
            var newAxis = permt[oldAxis];

            var newCon = Concat(new IR.Tuple((from input in wcinputs select result.GetExpr(input)).ToArray()), newAxis);
            var newTran = Transpose(newCon, oldPerm);
            return newTran;
        }

    }

    // public class TransPosePadMotion : EGraphRule
    // {
    //     ID wcpad = "pad", wctran = "tran", wcperm = "perm";
    //     public override ExprPattern GetPattern()
    //     {
    //         Transpose(Pad(), IsConst("perm"));
    //     }
    // }

}