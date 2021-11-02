using System;
using Nncase.IR;
using Nncase.Transform;
using Nncase.Transform.Pattern;
using static Nncase.Transform.Pattern.Utility;
namespace Nncase.Transform.Rule
{

    public abstract class EGraphRule
    {
        public virtual ExprPattern[] GetPatterns()
        {
            return new ExprPattern[] { Pattern };
        }

        public ExprPattern Pattern { get; set; };


        public virtual Expr? GetRePlace(EMatchResult result) => throw new NotImplementedException("Not Implement GetRePlace!");
    }

    public sealed class Reassociate : EGraphRule
    {
        private WildCardPattern wx = "x", wy = "y", wz = "z";

        public Reassociate()
        {

            Pattern = (wx * wy) * wz;
        }


        public override Expr GetRePlace(EMatchResult result)
        {
            Expr x = result.GetExpr(wx), y = result.GetExpr(wy), z = result.GetExpr(wz);
            return x * (y * z);
        }
    }

    public sealed class RemoveNoSenceAddSub : EGraphRule
    {

        private bool CheckScalarIsZero(DataType dataType, IRBytes bytes)
        => DataTypes.ToScalar<float>(dataType, bytes) == 0.0f;


        private bool CheckTensorIsZero(DataType dataType, IRBytes bytes)
        {
            throw new NotImplementedException("Not Implement For Check Tensor Is Zero.");
        }

        private readonly WildCardPattern[] wcs = new WildCardPattern[] { IsWildCard() };

        public RemoveNoSenceAddSub()
        {

            Pattern = IsBinary(x => x is (BinaryOp.Add or BinaryOp.Sub), wcs[0], IsConst(
              (Const x) => (x.ValueType, x.Data) switch
              {
                  (TensorType type, _) => type.IsScalar switch
                  {
                      true => CheckScalarIsZero(type.DType, x.Data),
                      false => CheckTensorIsZero(type.DType, x.Data)
                  },
                  (_, _) => false
              }
            ));
        }

        public override Expr GetRePlace(EMatchResult result)
        {
            return result.GetExpr(wcs[0]);
        }

    }

}