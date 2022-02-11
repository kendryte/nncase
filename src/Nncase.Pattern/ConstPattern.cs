// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.IR;
using static Nncase.IR.Utility;

namespace Nncase.Pattern;

public sealed record ConstPattern(Func<Const, bool> Cond) : ExprPattern
{
    /// <summary>
    /// <see cref="Target"/>.
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

    public bool MatchLeaf(Const expr)
    {
        return Cond(expr) && MatchCheckedType(expr);
    }
}

public static partial class Utility
{
    public static ConstPattern IsConst() => new ConstPattern(x => x is Const);

    public static ConstPattern IsConst(Func<Const, bool> Cond) => new ConstPattern(Cond);

    public static TensorConstPattern IsConst(Func<float, bool> cond) => new TensorConstPattern(
      x =>
      {
          if (DataTypes.IsFloat(x.ValueType.DType))
          {
              if (x.ValueType.IsScalar)
              {
                  return cond(x.Value.ToScalar<float>());
              }
              else
              {
                  return x.Value.Cast<float>().All(cond);
              }
          }

          return false;
      });

    public static TensorConstPattern IsConst(Func<int, bool> cond) => new TensorConstPattern(
      x =>
      {
          if (DataTypes.IsIntegral(x.ValueType.DType))
          {
              if (x.ValueType.IsScalar)
              {
                  return cond(x.Value.ToScalar<int>());
              }
              else
              {
                  return x.Value.Cast<int>().All(cond);
              }
          }

          return false;
      });

    public static ConstPattern IsConst(TypePattern typePattern) => new ConstPattern(x => typePattern.MatchLeaf(x.ValueType));

    public static TensorConstPattern IsConstIntTensor() => IsTensorConst(IsIntegral());

    public static TensorConstPattern IsConstIntSclar() => IsTensorConst(IsIntegral());

    public static TensorConstPattern IsConst<T>(T Value)
    where T : unmanaged, IEquatable<T>
    => new TensorConstPattern(x => x == Const.FromScalar(Value));
}
