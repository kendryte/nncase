using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using Nncase.Pattern;
using Nncase.Pattern.Tensors;
using Nncase.Pattern.Math;
using static Nncase.Pattern.F.Math;
using static Nncase.Pattern.F.Tensors;
using static Nncase.Pattern.Utility;
using static Nncase.IR.F.Math;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.Utility;
using Nncase.Evaluator;
using TorchSharp;

namespace Nncase.Transform.Rule
{

    /// <summary>
    /// Motion Transpose with Binary
    /// binary(transpose(a,p),transpose(b,p)) => transpose(binary(a,b),p)
    /// </summary>
    /// 
    public class TransposeBinaryMotion : PatternRule
    {
        private ConstPattern conpat;
        private TransposeWrapper transRhs;
        private TransposeWrapper transLhs;
        private BinaryWrapper binary;

        /// <summary>
        /// <see cref="TransposeBinaryMotion"/>
        /// </summary>
        public TransposeBinaryMotion()
        {
            conpat = IsConst();
            transLhs = Transpose(IsWildCard(), conpat);
            transRhs = Transpose(IsWildCard(), conpat);
            binary = IsBinary(transLhs, transRhs);
            Pattern = binary;
        }

        /// <inheritdoc/>
        public override Expr GetRePlace(IMatchResult result)
        {
            transRhs.Bind(result);
            transLhs.Bind(result);
            binary.Bind(result);
            return Transpose(Binary(binary.BinaryOp, transLhs.Input(), transRhs.Input()), result[conpat]);
        }
    }

    /// <summary>
    /// binary(transpose(a, p), const) =>  transpose(binary(a, const), p)
    /// </summary>
    public class TransposeConstBinaryMotionLeft : PatternRule
    {
        TransposeWrapper transpose = Transpose(IsWildCard(), IsConst());
        ConstPattern con = IsConst();

        BinaryWrapper binary;

        /// <summary>
        /// <see cref="TransposeConstBinaryMotionLeft"/>
        /// </summary>
        public TransposeConstBinaryMotionLeft()
        {
            Pattern = binary = IsBinary(transpose, con);
        }

        /// <inheritdoc/>
        public override Expr? GetRePlace(IMatchResult result)
        {
            binary.Bind(result);
            transpose.Bind(result);
            var newCon = GetNewConst(result[con], transpose.Perm<Const>());
            return Transpose(Binary(binary.BinaryOp, transpose.Input(), newCon), transpose.Perm());
        }

        /// <summary>
        /// Get the permed const with new shape
        /// </summary>
        /// <param name="oldCon"></param>
        /// <param name="oldPerm"></param>
        /// <returns> new const expr</returns>
        public static Const GetNewConst(Const oldCon, Const oldPerm)
        {
            var perm = oldPerm.ToTensor<int>();
            var new_perm = perm.Select((p, i) => (p, i)).OrderBy(k => k.p).Select(k => (long)k.i);
            var ts = oldCon.ToTorchTensor();
            return torch.permute(ts, new_perm.ToArray()).ToConst();
        }
    }
    /// <summary>
    /// binary(const, transpose(a, p)) =>  transpose(binary(const, a), p)
    /// </summary>
    public class TransposeConstBinaryMotionRight : PatternRule
    {
        TransposeWrapper transpose = Transpose(IsWildCard(), IsConst());
        ConstPattern con = IsConst();

        BinaryWrapper binary;
        /// <summary>
        /// <see cref="TransposeConstBinaryMotionRight"/>
        /// </summary>
        public TransposeConstBinaryMotionRight()
        {
            Pattern = binary = IsBinary(con, transpose);
        }

        /// <inheritdoc/>
        public override Expr GetRePlace(IMatchResult result)
        {
            binary.Bind(result);
            transpose.Bind(result);
            var newCon = TransposeConstBinaryMotionLeft.GetNewConst(result[con], transpose.Perm<Const>());
            return Transpose(Binary(binary.BinaryOp, newCon, transpose.Input()), transpose.Perm());
        }
    }

    public class TransposeConcatMotion : PatternRule
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

        public override Expr? GetRePlace(IMatchResult result)
        {
            var newShapes = (from input in wcinputs select GetShape(result[input])).ToArray();
            var oldPerm = result.GetExpr<Const>(wcprem);
            var permt = oldPerm.ToTensor<int>();
            var oldAxis = result.GetExpr<Const>(wcaxis).ToScalar<int>();
            var newAxis = permt[oldAxis];

            var newCon = Concat(new IR.Tuple((from input in wcinputs select result[input]).ToArray()), newAxis);
            var newTran = Transpose(newCon, oldPerm);
            return newTran;
        }

    }

    public class TransPosePadMotion : PatternRule
    {
        WildCardPattern wcin = "input";

        ConstPattern wcmode = IsConst(IsScalar()), wcpadv = IsConst(IsScalar()), wcperm = IsConst(IsScalar()), wcpads = IsConst(IsTensor() & IsIntegral());

        CallPattern wcpad;

        public TransPosePadMotion()
        {
            wcpad = IsPad(wcin, wcpads, wcpadv);
            Pattern = Transpose(wcpad, wcperm);
        }

        public override Expr GetRePlace(IMatchResult result)
        {
            var input = result[wcin];
            var (mode, padv, perm) = result[wcmode, wcpadv, wcperm];
            var padst = result[wcpads].ToTensor<int>();
            var newpadspan = new int[padst.Dimensions[0] * 2];
            var permt = perm.ToTensor<int>();
            for (int i = 0; i < permt.Dimensions[0]; i++)
            {
                newpadspan[i * 2] = padst[permt[i], 0];
                newpadspan[(i * 2) + 1] = padst[permt[i], 1];
            }
            Const newPads = new Const(result[wcpads].ValueType, newpadspan.Cast<byte>().ToArray());
            return Pad(Transpose(input, perm), (newPads), ((Pad)result[wcpad].Target).PadMode, padv);
        }
    }

    public class TransposeReduceMotion : PatternRule
    {

        WildCardPattern wcinput = "input", wcinit = "init";
        ConstPattern wckeepdims = IsConst(IsScalar() | IsIntegral());
        WildCardPattern wcaxis = "axis", wcperm = "axis";

        public TransposeReduceMotion()
        {
            Pattern = IsReduce(Transpose(wcinput, wcperm), wcaxis, wcinit, wckeepdims);
        }

        public override Expr? GetRePlace(IMatchResult result)
        {
            var (input, axis, init) = result[wcinput, wcaxis, wcinit];
            var perm = result[wcperm];
            var keepdims = result[wckeepdims].ToScalar<bool>();
            var reduce = result.GetRoot<Reduce>();
            var new_axis = Gather(perm, 0, axis);
            if (keepdims == false)
            {
                return Squeeze(Transpose(Reduce(reduce.ReduceOp, input, new_axis, init, true), perm), axis);
            }
            return Transpose(Reduce(reduce.ReduceOp, input, new_axis, init, keepdims), perm);
        }
    }

    public class TransposeUnaryMotion : PatternRule
    {

        WildCardPattern wcinput = "input", wcperm = "perm";

        CallPattern wcunary;
        public TransposeUnaryMotion()
        {
            wcunary = IsUnary(wcinput);
            Pattern = Transpose(wcunary, wcperm);
        }

        public override Expr? GetRePlace(IMatchResult result)
        {
            var (input, perm) = result[wcinput, wcperm];
            var unarycall = result[wcunary];
            var unaryop = ((Unary)unarycall.Target).UnaryOp;

            return Unary(unaryop, Transpose(input, perm));
        }
    }

    public class TransposeClampMotion : PatternRule
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

        public override Expr? GetRePlace(IMatchResult result)
        {
            var (min, max, perm) = result[wcmin, wcmax, wcperm];
            var input = result[wcinput];
            return Transpose(Clamp(input, min, max), perm);
        }

    }

}