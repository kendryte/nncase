using System;
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
                last_const_ ??= con;
                return last_const_ == con;
            }
        }
        private Func<Const, bool> comp = new(new Comparer().Cond);

        ID wcvargs, wcaxis;

        public TransposeConcatMotion()
        {
            // wcperm = new WildCardPattern(GetID(), IsConst(comp));
            // wcinputs = IsWildCard(GetID());
            // wcvargs = IsWildCard("wcvargs", Transpose(wcinputs, wcperm));
        }

        public override ExprPattern GetPattern()
        {
            return Concat(IsTuple(IsVArgsRepeat(n =>
            {
                var pats = new ExprPattern[n];
                for (int i = 0; i < n; i++)
                {
                    pats[i] = Transpose(IsWildCard(), IsConst(comp));
                }
                return pats;
            })), IsConst(wcaxis));
        }

        public override Expr GetRePlace(EMatchResult result)
        {
            // Expr inputs = result.Context[wcin].Expr, axis = result.Context[wcaxis].Expr;
            return result.GetRoot();
        }

    }

}