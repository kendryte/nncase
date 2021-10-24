using System;
using Nncase.IR;
using Nncase.Transform;
using Nncase.Transform.Pattern;
using static Nncase.Transform.Pattern.Functional;
namespace Nncase.Transform.Rule
{

    public interface IEGraphRule
    {
        public ExprPattern GetPattern();
        public Expr GetRePlace(EMatchResult result);
    }

    public class Reassociate : IEGraphRule
    {
        private WildCardPattern wx = "x", wy = "y", wz = "z";

        public ExprPattern GetPattern()
        {
            return (wx * wy) * wz;
        }

        public Expr GetRePlace(EMatchResult result)
        {
            Expr x = result.Context[wx].Expr, y = result.Context[wy].Expr, z = result.Context[wz].Expr;
            return x * (y * z);
        }
    }

    public class RemoveNoSenceAddSub : IEGraphRule
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

        private readonly WildCardPattern[] wcs = new WildCardPattern[] { WildCard() };

        public ExprPattern GetPattern()
        {
            return IsBinary(x => x is (BinaryOp.Add or BinaryOp.Sub), wcs[0], IsConst(
              (Const x) => (x.ValueType, x.Data) switch
              {
                  (TensorType type, _) => type.IsScalar switch
                  {
                      true => CheckScalarIsZero(type.DataType, x.Data),
                      false => CheckTensorIsZero(type.DataType, x.Data)
                  },
                  (_, _) => false
              }
            ));
        }

        public Expr GetRePlace(EMatchResult result)
        {
            return result.Context[wcs[0]].Expr;
        }

    }

}