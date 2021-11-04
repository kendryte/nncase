using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using Nncase.Transform.Pattern;
using static Nncase.Transform.Pattern.F.Math;
using static Nncase.Transform.Pattern.F.Tensors;
using static Nncase.Transform.Pattern.Utility;
using static Nncase.IR.F.Math;
using static Nncase.IR.F.Tensors;

namespace Nncase.Transform.Rule
{
    public class TransposeBinaryMotion : EGraphRule
    {
        private WildCardPattern wx = "x", wy = "y", wperm = "perm";
        private WildCardPattern wbin = new();
        public TransposeBinaryMotion()
        {
            Pattern = IsBinary(Transpose(wx, wperm), Transpose(wy, wperm));
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
        protected WildCardPattern wx = "x";
        protected ConstPattern wperm = IsConst(IsTensor()), wcon = IsConst();

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
            newCon = con with { ValueType = (TensorType)con.ValueType with { Shape = newShape } };
        }
    }

    public class TransposeConstantBinaryMotionLeft : TransposeConstantBinaryMotion
    {
        public TransposeConstantBinaryMotionLeft()
        {
            Pattern = IsBinary(Transpose(wx, wperm), wcon);
        }

        public override Expr GetRePlace(EMatchResult result)
        {
            ParserResult(result);
            return Transpose(new Call(new Binary(binary.BinaryOp), x, newCon), perm);
        }
    }

    public class TransposeConstantBinaryMotionRight : TransposeConstantBinaryMotion
    {
        public TransposeConstantBinaryMotionRight()
        {
            Pattern = IsBinary(wcon, Transpose(wx, wperm));
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
        List<WildCardPattern> wcinputs = new();
        ConstPattern wcprem, wcaxis;

        public TransposeConcatMotion()
        {
            wcprem = IsConst(IsTensor());
            wcaxis = IsConst(IsScalar());
            Pattern = Concat(IsTuple(IsVArgsRepeat((n, param) =>
              {
                  for (int i = 0; i < n; i++)
                  {
                      var wcin = IsWildCard();
                      param.Add(Transpose(wcin, wcprem));
                  }
              }
            )), wcaxis);
        }

        public override Expr? GetRePlace(EMatchResult result)
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

    public class TransPosePadMotion : EGraphRule
    {
        WildCardPattern wcin = "input";

        ConstPattern wcmode = IsConst(IsScalar()), wcpadv = IsConst(IsScalar()), wcperm = IsConst(IsScalar()), wcpads = IsConst(IsTensor() & IsIntegral());

        CallPattern wcpad;

        public TransPosePadMotion()
        {
            wcpad = IsPad(wcin, wcpads, wcpadv);
            Pattern = Transpose(wcpad, wcperm);
        }

        public override Expr GetRePlace(EMatchResult result)
        {
            var input = result.GetExpr(wcin);
            var (mode, padv, perm) = result.GetExpr(wcmode, wcpadv, wcperm);
            var padst = result.GetExpr(wcpads).ToTensor<int>();
            var newpadspan = new int[padst.Dimensions[0] * 2];
            var permt = perm.ToTensor<int>();
            for (int i = 0; i < permt.Dimensions[0]; i++)
            {
                newpadspan[i * 2] = padst[permt[i], 0];
                newpadspan[(i * 2) + 1] = padst[permt[i], 1];
            }
            Const newPads = new Const(result.GetExpr(wcpads).ValueType, newpadspan.Cast<byte>().ToArray());
            return Pad(Transpose(input, perm), (newPads), ((Pad)result.GetExpr(wcpad).Target).padMode, padv);
        }
    }

    public class TransposeReduceMotion : EGraphRule
    {

        WildCardPattern wcinput = "input", wcinit = "init";
        ConstPattern wckeepdims = IsConst(IsScalar() | IsIntegral());
        WildCardPattern wcaxis = "axis", wcperm = "axis";

        public TransposeReduceMotion()
        {
            Pattern = IsReduce(Transpose(wcinput, wcperm), wcaxis, wcinit, wckeepdims);
        }

        public override Expr? GetRePlace(EMatchResult result)
        {
            var (input, axis, init) = result.GetExpr(wcinput, wcaxis, wcinit);
            var perm = result.GetExpr(wcperm);
            var keepdims = result.GetExpr(wckeepdims).ToScalar<bool>();
            var reduce = result.GetRoot<Reduce>();
            var new_axis = Gather(perm, 0, axis);
            if (keepdims == false)
            {
                return Squeeze(Transpose(Reduce(reduce.reduceOp, input, new_axis, init, true), perm), axis);
            }
            return Transpose(Reduce(reduce.reduceOp, input, new_axis, init, keepdims), perm);
        }
    }

    public class TransposeUnaryMotion : EGraphRule
    {

        WildCardPattern wcinput = "input", wcperm = "perm";

        CallPattern wcunary;
        public TransposeUnaryMotion()
        {
            wcunary = IsUnary(wcinput);
            Pattern = Transpose(wcunary, wcperm);
        }

        public override Expr? GetRePlace(EMatchResult result)
        {
            var (input, perm) = result.GetExpr(wcinput, wcperm);
            var unarycall = result.GetExpr(wcunary);
            var unaryop = ((Unary)unarycall.Target).UnaryOp;

            return Unary(unaryop, Transpose(input, perm));
        }
    }

    public class TransposeClampMotion : EGraphRule
    {
        ConstPattern wcmin, wcmax, wcperm;
        WildCardPattern wcinput;
        TransposeClampMotion()
        {
            wcmin = IsConst(IsScalar());
            wcmax = IsConst(IsScalar());
            wcperm = IsConst(IsTensor() & IsIntegral());
            wcinput = IsWildCard();
            Pattern = Clamp(Transpose(wcinput, wcperm), wcmin, wcmax);
        }

        public override Expr? GetRePlace(EMatchResult result)
        {
            var (min, max, perm) = result.GetExpr(wcmin, wcmax, wcperm);
            var input = result.GetExpr(wcinput);
            return Transpose(Clamp(input, min, max), perm);
        }

    }

}