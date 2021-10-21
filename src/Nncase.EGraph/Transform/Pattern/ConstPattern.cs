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

    public static partial class Functional
    {

        public static ConstPattern IsConst() => new ConstPattern(x => x is Const);

        public static ConstPattern IsConst(Func<float, bool> cond) => new ConstPattern(
          x => x.ValueType switch
          {
              TensorType tensor => tensor.IsScalar && (tensor.DataType is (DataType.Float32 or DataType.Float64)) && cond(ToFloat(x.Data)),
              _ => false
          }
        );

        public static ConstPattern IsConst(Func<int, bool> cond) => new ConstPattern(
          x => x.ValueType switch
          {
              TensorType tensor => tensor.IsScalar && (tensor.DataType is (DataType.Int8 or DataType.Int16 or DataType.Int32 or DataType.Int64)) && cond(ToInt(x.Data)),
              _ => false
          }
        );

        private static float ToFloat(byte[] bytes) => bytes.Length switch
        {
            8 => (float)BitConverter.ToDouble(bytes),
            4 => (float)BitConverter.ToSingle(bytes),
            _ => throw new InvalidCastException($"Can't Cast Bytes Data To Float, Length is {bytes.Length}!")
        };

        private static int ToInt(byte[] bytes) => bytes.Length switch
        {
            8 => (int)BitConverter.ToInt64(bytes),
            4 => (int)BitConverter.ToInt32(bytes),
            2 => (int)BitConverter.ToInt16(bytes),
            1 => (int)bytes[0],
            _ => throw new InvalidCastException($"Can't Cast Bytes Data To Int, Length is {bytes.Length}!")
        };

    }

}