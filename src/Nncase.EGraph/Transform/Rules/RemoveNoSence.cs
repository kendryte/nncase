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
            return new ExprPattern[] { GetPattern() };
        }

        public virtual ExprPattern GetPattern() => throw new NotImplementedException("Not Implement GetPattern!");

        public virtual Expr GetRePlace(EMatchResult result) => throw new NotImplementedException("Not Implement GetRePlace!");
    }

    public sealed class Reassociate : EGraphRule
    {
        private WildCardPattern wx = "x", wy = "y", wz = "z";

        public override ExprPattern GetPattern()
        {
            return (wx * wy) * wz;
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
        =>
            dataType switch
            {
                (DataType.Int8 or DataType.Int16
                or DataType.Int32 or DataType.Int64)
                => ToInt(dataType, bytes) == 0,
                (DataType.UInt8 or DataType.UInt16
                or DataType.UInt32 or DataType.UInt64)
                => ToUInt(dataType, bytes) == 0,
                (DataType.BFloat16 or DataType.Float16
                or DataType.Float32 or DataType.Float64)
                => ToFloat(dataType, bytes) == 0.0f,
                _ => ToFloat(dataType, bytes) == 0.0f
            };


        private bool CheckTensorIsZero(DataType dataType, IRBytes bytes)
        {
            throw new NotImplementedException("Not Implement For Check Tensor Is Zero.");
        }

        private readonly WildCardPattern[] wcs = new WildCardPattern[] { IsWildCard() };

        public override ExprPattern GetPattern()
        {
            return IsBinary(x => x is (BinaryOp.Add or BinaryOp.Sub), wcs[0], IsConst(
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