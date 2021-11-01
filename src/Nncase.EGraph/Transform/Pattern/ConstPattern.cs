using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.IR;

namespace Nncase.Transform.Pattern
{

    public sealed record ConstPattern(Func<Const, bool> Cond) : ExprPattern
    {
        public ConstPattern(Const expr) : this(x => x == expr) { }

        public static implicit operator ConstPattern(byte value) => new ConstPattern((Const)value);

        public static implicit operator ConstPattern(ushort value) => new ConstPattern((Const)value);

        public static implicit operator ConstPattern(uint value) => new ConstPattern((Const)value);

        public static implicit operator ConstPattern(ulong value) => new ConstPattern((Const)value);

        public static implicit operator ConstPattern(sbyte value) => new ConstPattern((Const)value);

        public static implicit operator ConstPattern(short value) => new ConstPattern((Const)value);

        public static implicit operator ConstPattern(int value) => new ConstPattern((Const)value);

        public static implicit operator ConstPattern(long value) => new ConstPattern((Const)value);

        public static implicit operator ConstPattern(Half value) => new ConstPattern((Const)value);

        public static implicit operator ConstPattern(float value) => new ConstPattern((Const)value);

        public static implicit operator ConstPattern(double value) => new ConstPattern((Const)value);

        public static implicit operator ConstPattern(BFloat16 value) => new ConstPattern((Const)value);

        public static implicit operator ConstPattern(bool value) => new ConstPattern((Const)value);


        public bool MatchLeaf(Const expr)
        {
            return Cond(expr) && MatchCheckedType(expr);
        }
    }

    public static partial class Utility
    {

        public static ConstPattern IsConst() => new ConstPattern(x => x is Const);


        public static ConstPattern IsConst(Func<Const, bool> Cond) => new ConstPattern(Cond);

        public static ConstPattern IsConst(Func<float, bool> cond) => new ConstPattern(
          x => x.ValueType switch
              {

                  TensorType tensor => tensor.IsScalar &&
                   (tensor.DType is (DataType.BFloat16 or DataType.Float16 or DataType.Float32 or DataType.Float64)) &&
                       cond(DataTypes.ToScalar<float>(tensor.DType, x.Data)),
                  _ => false
              }
            );


        public static ConstPattern IsConst(Func<int, bool> cond) => new ConstPattern(
          x => x.ValueType switch
          {
              TensorType tensor => tensor.IsScalar &&
                (tensor.DType is (DataType.UInt8 or DataType.UInt16 or DataType.UInt32 or DataType.UInt64 or
                                  DataType.Int8 or DataType.Int16 or DataType.Int32 or DataType.Int64))
                  && cond(DataTypes.ToScalar<int>(tensor.DType, x.Data)),
              _ => false
          }
        );

        public static ConstPattern IsConstTensor() => new ConstPattern(
          x => x.ValueType switch
          {
              TensorType tensor => tensor.IsTensor,
              _ => false
          }
        );

        public static ConstPattern IsConstScalar() => new ConstPattern(
          x => x.ValueType switch
          {
              TensorType tensor => tensor.IsScalar,
              _ => false
          }
        );

        public static ConstPattern IsConst<T>(T Value)
        where T : unmanaged
        => new ConstPattern(x => x == Const.FromScalar<T>(Value));



    }

}