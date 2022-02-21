// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.IR;
using static Nncase.IR.TypePatternUtility;

namespace Nncase.PatternMatch;

/// <summary>
/// Pattern for <see cref="Const"/>.
/// </summary>
/// <param name="Condition">Expression condition.</param>
public sealed record ConstPattern(Func<Const, bool> Condition) : Pattern<Const>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="ConstPattern"/> class.
    /// </summary>
    /// <param name="const"><see cref="Const"/> expression.</param>
    public ConstPattern(Const @const)
        : this(x => x.Equals(@const))
    {
        Value = @const;
    }

    /// <summary>
    /// Gets value.
    /// </summary>
    public Const? Value { get; }

    /// <inheritdoc/>
    protected override bool MatchLeafCore(Const expr) => Condition(expr);
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
    => new TensorConstPattern(x => x == Tensor.FromScalar(Value));
}
