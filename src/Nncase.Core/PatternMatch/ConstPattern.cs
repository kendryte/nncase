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
/// <param name="Name">name.</param>
public sealed record ConstPattern(Func<Const, bool> Condition, string? Name) : Pattern<Const>(Name)
{
    /// <summary>
    /// Initializes a new instance of the <see cref="ConstPattern"/> class.
    /// </summary>
    /// <param name="const"><see cref="Const"/> expression.</param>
    /// <param name="name">name.</param>
    public ConstPattern(Const @const, string? name)
        : this(x => x.Equals(@const), name)
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
    /// <summary>
    /// create const pattern.
    /// </summary>
    /// <param name="name">name.</param>
    /// <returns>ConstPattern.</returns>
    public static ConstPattern IsConst(string? name) => new(x => x is not null, name);

    public static ConstPattern IsConst() => IsConst(name: null);

    /// <summary>
    /// create const pattern.
    /// </summary>
    /// <param name="name">name.</param>
    /// <param name="cond">condition.</param>
    /// <returns>ConstPattern.</returns>
    public static ConstPattern IsConst(string? name, Func<Const, bool> cond) => new(cond, name);

    public static ConstPattern IsConst(Func<Const, bool> cond) => IsConst(null, cond);

    /// <summary>
    /// create const pattern.
    /// </summary>
    /// <param name="name">name.</param>
    /// <param name="cond">condition.</param>
    /// <returns>ConstPattern.</returns>
    public static TensorConstPattern IsConst(string? name, Func<float, bool> cond) => new(
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
      },
      name);

    public static TensorConstPattern IsConst(Func<float, bool> cond) => IsConst(null, cond);

    /// <summary>
    /// create const pattern.
    /// </summary>
    /// <param name="name">name.</param>
    /// <param name="cond">condition.</param>
    /// <returns>ConstPattern.</returns>
    public static TensorConstPattern IsConst(string? name, Func<int, bool> cond) => new(
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
      },
      name);

    public static TensorConstPattern IsConst(Func<int, bool> cond) => IsConst(null, cond);

    /// <summary>
    /// create const pattern.
    /// </summary>
    /// <param name="name">name.</param>
    /// <param name="typePattern">type pattern.</param>
    /// <returns>ConstPattern.</returns>
    public static ConstPattern IsConst(string? name, TypePattern typePattern) => new(x => typePattern.MatchLeaf(x.ValueType), name);

    public static ConstPattern IsConst(TypePattern typePattern) => IsConst(typePattern);

    /// <summary>
    /// create const pattern.
    /// </summary>
    /// <param name="name">name.</param>
    /// <returns>ConstPattern.</returns>
    public static TensorConstPattern IsConstIntTensor(string? name) => IsTensorConst(name, IsIntegral());

    public static TensorConstPattern IsConstIntTensor() => IsConstIntTensor(null);

    /// <summary>
    /// create const pattern.
    /// </summary>
    /// <param name="name">name.</param>
    /// <returns>ConstPattern.</returns>
    public static TensorConstPattern IsConstIntSclar(string? name) => IsTensorConst(name, IsIntegral());

    public static TensorConstPattern IsConstIntSclar() => IsTensorConst(null, IsIntegral());

    /// <summary>
    /// create const pattern.
    /// </summary>
    /// <typeparam name="T">target value type.</typeparam>
    /// <param name="name">name.</param>
    /// <param name="value">value.</param>
    /// <returns>ConstPattern.</returns>
    public static TensorConstPattern IsConst<T>(string? name, T value)
        where T : unmanaged, IEquatable<T>
    => new(x => x.Value is Tensor<T> { Length: 1 } t && EqualityComparer<T>.Default.Equals(t[0], value), name);

    public static TensorConstPattern IsConst<T>(T value)
        where T : unmanaged, IEquatable<T> => IsConst<T>(null, value);
}
