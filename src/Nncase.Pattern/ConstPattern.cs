using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.IR;
using static Nncase.IR.TypePatternUtility;

namespace Nncase.Pattern
{

    public sealed record ConstPattern(Func<Const, bool> Cond) : ExprPattern
    {
        /// <summary>
        /// <see cref="Target"/>
        /// </summary>
        private readonly Const? _target = null;

        /// <summary>
        /// save the target const for match, we can print it for debug.
        /// </summary>
        public Const? Target { get => _target; }

        public ConstPattern(Const expr) : this(x => x == expr)
        {
            _target = expr;
        }

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
          x =>
          {
              if (DataTypes.IsFloat(x.ValueType.DType))
              {
                  if (x.ValueType.IsScalar)
                      return cond(x.ToScalar<float>());
                  else
                      return x.ToTensor<float>().All(cond);
              }
              return false;
          });

        public static ConstPattern IsConst(Func<int, bool> cond) => new ConstPattern(
          x =>
          {
              if (DataTypes.IsIntegral(x.ValueType.DType))
              {
                  if (x.ValueType.IsScalar)
                      return cond(x.ToScalar<int>());
                  else
                      return x.ToTensor<int>().All(cond);
              }
              return false;
          });

        public static ConstPattern IsConst(TypePattern typePattern) => new ConstPattern(x => typePattern.MatchLeaf(x.ValueType));

        public static ConstPattern IsConstIntTensor() => IsConst(IsTensor() & IsIntegral());
        public static ConstPattern IsConstIntSclar() => IsConst(IsScalar() & IsIntegral());

        public static ConstPattern IsConst<T>(T Value)
        where T : unmanaged
        => new ConstPattern(x => x == Const.FromScalar<T>(Value));
    }

}