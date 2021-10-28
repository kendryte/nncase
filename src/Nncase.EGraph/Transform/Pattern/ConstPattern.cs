using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.IR;

namespace Nncase.Transform.Pattern
{

    public sealed record ConstPattern(ID Id, Func<Const, bool> Cond) : ExprPattern(Id)
    {
        public ConstPattern(Const expr) : this(Utility.GetID(), x => x == expr) { }

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

        public static ConstPattern IsConst(ID Id) => new ConstPattern(Id, x => x is Const);

        public static ConstPattern IsConst() => IsConst(GetID());

        public static ConstPattern IsConst(ID Id, Func<Const, bool> Cond) => new ConstPattern(Id, Cond);
        public static ConstPattern IsConst(Func<Const, bool> Cond) => new ConstPattern(GetID(), Cond);

        public static ConstPattern IsConst(ID Id, Func<float, bool> cond) => new ConstPattern(Id,
          x => x.ValueType switch
          {
              TensorType tensor => tensor.IsScalar && (tensor.DType is (DataType.Float32 or DataType.Float64)) && cond(ToFloat(tensor.DType, x.Data)),
              _ => false
          }
        );

        public static ConstPattern IsConst(Func<float, bool> cond) => IsConst(GetID(), cond);

        public static ConstPattern IsConst(ID Id, Func<int, bool> cond) => new ConstPattern(Id,
          x => x.ValueType switch
          {
              TensorType tensor => tensor.IsScalar && (tensor.DType is (DataType.Int8 or DataType.Int16 or DataType.Int32 or DataType.Int64)) && cond(ToInt(tensor.DType, x.Data)),
              _ => false
          }
        );

        public static ConstPattern IsConst(Func<int, bool> cond) => IsConst(GetID(), cond);


        public static ConstPattern IsConstTensor(ID Id) => new ConstPattern(Id,
          x => x.ValueType switch
          {
              TensorType tensor => tensor.IsTensor,
              _ => false
          }
        );
        public static ConstPattern IsConstTensor() => IsConstTensor(GetID());

        public static ConstPattern IsConstScalar(ID Id) => new ConstPattern(Id,
          x => x.ValueType switch
          {
              TensorType tensor => tensor.IsScalar,
              _ => false
          }
        );

        public static ConstPattern IsConstScalar() => IsConstScalar(GetID());

        public static ConstPattern IsConst<T>(ID Id, T Value)
        where T : unmanaged
        => new ConstPattern(Id, x => x == Const.FromScalar<T>(Value));

        public static ConstPattern IsConst<T>(T Value)
        where T : unmanaged
        => new ConstPattern(GetID(), x => x == Const.FromScalar<T>(Value));

        public static int ToInt(DataType dataType, byte[] bytes) => dataType switch
        {
            DataType.Int64 => (int)BitConverter.ToInt64(bytes),
            DataType.Int32 => (int)BitConverter.ToInt32(bytes),
            DataType.Int16 => (int)BitConverter.ToInt16(bytes),
            DataType.Int8 => (int)bytes[0],
            _ => throw new InvalidCastException($"Can't Cast Bytes Data")
        };
        public static uint ToUInt(DataType dataType, byte[] bytes) =>
        dataType switch
        {
            DataType.UInt64 => (uint)BitConverter.ToUInt64(bytes),
            DataType.UInt32 => (uint)BitConverter.ToUInt32(bytes),
            DataType.UInt16 => (uint)BitConverter.ToUInt16(bytes),
            DataType.UInt8 => (uint)bytes[0],
            _ => throw new InvalidCastException($"Can't Cast Bytes Data")
        };

        public static float ToFloat(DataType dataType, byte[] bytes) =>
        dataType switch
        {
            DataType.Float64 => (float)BitConverter.ToDouble(bytes),
            DataType.Float32 => (float)BitConverter.ToSingle(bytes),
            DataType.BFloat16 => (float)(new BFloat16(bytes[0])),
            _ => throw new InvalidCastException($"Can't Cast Bytes Data")
        };

    }

}