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
            Expr x = result.Context[wx].Expr, y = result.Context[wy].Expr, perm = result.Context[wperm].Expr;
            var bin = (Binary)result.Root.Expr;
            return Transpose(new Call(new Binary(bin.BinaryOp), x, y), perm);
        }
    }

    public class TransposeConstantBinaryMotion : EGraphRule
    {
        protected WildCardPattern wx = "x", wperm = "wperm";
        protected WildCardPattern wcon = new WildCardPattern("wcon", IsConstTensor());

        // protected virtual CheckPerm()
        // {
        //     // pass
        // }

    }

    public class TransposeConstantBinaryMotionLeft : TransposeConstantBinaryMotion
    {
        public override ExprPattern GetPattern()
        {
            return IsBinary(Transpose(wx, wperm), wcon);
        }

        public override Expr GetRePlace(EMatchResult result)
        {
            Expr x = result.Context[wx].Expr, con = result.Context[wcon].Expr, perm = result.Context[wperm].Expr;
            var bin = (Binary)result.Root.Expr;
            return Transpose(new Call(new Binary(bin.BinaryOp), x, con), perm);
        }
    }

    public class TransposeConstantBinaryMotionRight : TransposeConstantBinaryMotion
    {
        public override ExprPattern GetPattern()
        {
            return IsBinary(wcon, Transpose(wx, wperm));
        }

        public override Expr GetRePlace(EMatchResult result)
        {
            Expr x = result.Context[wx].Expr, con = result.Context[wcon].Expr, perm = result.Context[wperm].Expr;
            var bin = (Binary)result.Root.Expr;
            return Transpose(new Call(new Binary(bin.BinaryOp), con, x), perm);
        }
    }


    // public class TransposeConcatMotion : EGraphRule
    // {

    //     public override ExprPattern GetPattern()
    //     {
    //         // return 
    //         // return IsAnyType;
    //     }

    //     public override Expr GetRePlace(EMatchResult result)
    //     {
    //         return base.GetRePlace(result);
    //     }

    // }

}