// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Runtime.CompilerServices;

namespace Nncase.IR;

public record TypePattern(Func<IRType, bool> Cond, string Reason)
{
    public TypePattern(IRType valueType)
        : this(x => x == valueType, $"Type = {valueType.ToString()}")
    {
    }

    public TypePattern(AnyType valueType)
        : this(x => x == valueType, $"Type = {valueType.ToString()}")
    {
    }

    public TypePattern(TensorType valueType)
        : this(x => x == valueType, $"Type = {valueType.ToString()}")
    {
    }

    public TypePattern(InvalidType valueType)
        : this(x => x == valueType, $"Type = {valueType.ToString()}")
    {
    }

    public TypePattern(TupleType valueType)
        : this(x => x == valueType, $"Type = {valueType.ToString()}")
    {
    }

    public TypePattern(CallableType valueType)
        : this(x => x == valueType, $"Type = {valueType.ToString()}")
    {
    }

    public static TypePattern operator &(TypePattern lhs, TypePattern rhs) => new TypePattern(x => lhs.Cond(x) && rhs.Cond(x), $"<{lhs.Reason}> And <{rhs.Reason}>");

    public static TypePattern operator |(TypePattern lhs, TypePattern rhs) => new TypePattern(x => lhs.Cond(x) || rhs.Cond(x), $"<{lhs.Reason}> Or <{rhs.Reason}>");

    public static TypePattern operator !(TypePattern lhs) => new TypePattern(x => !lhs.Cond(x), $"<Not {lhs.Reason}>");

    public bool MatchLeaf(IRType valueType) => Cond(valueType);

    /// <summary>
    /// Check the irtype, if not equal, throw exception.
    /// </summary>
    /// <param name="valueType">give the ir type.</param>
    /// <param name="fieldName"> the argument name.</param>
    public T Check<T>(T valueType, string fieldName)
        where T : IRType
    {
        if (valueType is TensorType { Shape: { IsUnranked: true } } || valueType is DistributedType { TensorType: { Shape: { IsUnranked: true } } })
        {
            return valueType;
        }

        if (valueType == null || (valueType is TensorType t && !MatchLeaf(t)) || (valueType is DistributedType d && !MatchLeaf(d.TensorType)))
        {
            var cur = valueType is null ? "None" : CompilerServices.Print(valueType);
            throw new InvalidOperationException($"{fieldName} Requrie <{Reason}>, But {cur}!");
        }

        return valueType;
    }
}

public static partial class TypePatternUtility
{
    /// <summary>
    /// the datatype is AnyType.
    /// <remarks>
    /// NOTE it's mean the tensorType/TupleType is error
    /// </remarks>
    /// </summary>
    public static TypePattern IsAnyType() => new TypePattern(AnyType.Default);

    /// <summary>
    /// custom the type cond.
    /// </summary>
    public static TypePattern IsType(Func<IRType, bool> typeCond, [CallerArgumentExpression("typeCond")] string reason = null!) => new TypePattern(typeCond, reason);

    /// <summary>
    /// this type is equal to target IRType.
    /// </summary>
    public static TypePattern IsType(IRType type) => IsType(x => x == type, $"Type = {type.ToString()}");

    /// <summary>
    /// is IsIRType.
    /// </summary>
    public static TypePattern IsIRType() => IsType(t => true, "IsIRType");

    /// <summary>
    /// is shape.
    /// </summary>
    public static TypePattern HasShape(Func<Shape, bool> shapeCond, string reason) => new TypePattern(
        x => x switch
        {
            TensorType ttype => shapeCond(ttype.Shape),
            _ => false,
        },
        reason);

    /// <summary>
    /// the tensor has FixedShape.
    /// </summary>
    /// <returns>TypePattern.</returns>
    public static TypePattern HasFixedShape() => HasShape(shape => shape.IsFixed, "HasFixedShape");

    /// <summary>
    /// the tensor has FixedShape.
    /// </summary>
    /// <returns>TypePattern.</returns>
    public static TypePattern HasRank() => HasShape(shape => shape.IsRanked, "HasRank");

    /// <summary>
    /// is target shape.
    /// </summary>
    public static TypePattern HasShape(Shape target_shape) => HasShape(
      inshape =>
        inshape.Rank == target_shape.Rank &&
        inshape.Zip(target_shape).All(
          (dim) => dim.Second == Dimension.Unknown ? true : dim.Second == dim.First),
      $"Shape = {target_shape}");

    /// <summary>
    /// is custom rank.
    /// </summary>
    public static TypePattern HasRank(Func<int, bool> cond, string reason) => HasShape(
      inshape => cond(inshape.Rank), reason);

    /// <summary>
    /// is target rank.
    /// </summary>
    public static TypePattern HasRank(int rank) => HasRank(r => r == rank, $"Rank = {rank}");

    /// <summary>
    /// is tensor.
    /// </summary>
    public static TypePattern IsTensor() => new TypePattern(
      x => x switch
      {
          TensorType ttype => ttype.IsTensor,
          _ => false,
      },
      "IsTensor");

    /// <summary>
    /// check the tensor is handle.
    /// </summary>
    public static TypePattern IsPointer() => new TypePattern(
      x => x is TensorType { IsScalar: true, DType: PointerType }, "IsPointer");

    /// <summary>
    /// check the datatype is scalar.
    /// </summary>
    public static TypePattern IsScalar() => new TypePattern(
      x => x switch
      {
          TensorType ttype => ttype.IsScalar,
          _ => false,
      },
      "IsScalar");

    /// <summary>
    /// check the datatype is integral.
    /// </summary>
    public static TypePattern HasDataType(DataType dataType) => new TypePattern(
      x => x switch
      {
          TensorType ttype => ttype.DType == dataType,
          _ => false,
      },
      $"{dataType.GetDisplayName()}");

    /// <summary>
    /// check the data type IsIntegral.
    /// </summary>
    public static TypePattern IsIntegral() => new TypePattern(
      x => x switch
      {
          TensorType ttype => DataTypes.IsIntegral(ttype.DType),
          DistributedType distributedType => DataTypes.IsIntegral(distributedType.TensorType.DType),
          _ => false,
      },
      "IsIntegral");

    /// <summary>
    /// chenck the data type isfloat.
    /// </summary>
    public static TypePattern IsFloat() => new TypePattern(
      x => x switch
      {
          TensorType ttype => DataTypes.IsFloat(ttype.DType),
          _ => false,
      },
      "IsFloat");

    /// <summary>
    /// check the data type is bool.
    /// </summary>
    public static TypePattern IsBool() => new TypePattern(
        x => x switch
        {
            TensorType ttype => ttype.DType == DataTypes.Boolean,
            _ => false,
        },
        "IsBool");

    /// <summary>
    /// int scalar.
    /// </summary>
    public static TypePattern IsIntegralScalar() => IsScalar() & IsIntegral();

    /// <summary>
    /// bool scalar.
    /// </summary>
    public static TypePattern IsBoolScalar() => IsScalar() & IsBool();

    /// <summary>
    /// float scalar.
    /// </summary>
    public static TypePattern IsFloatScalar() => IsScalar() & IsFloat();

    /// <summary>
    /// is tuple type.
    /// </summary>
    /// <param name="cond">the conditions.</param>
    /// <param name="reason"> reason. </param>
    /// <returns>TypePattern.</returns>
    public static TypePattern IsTuple(Func<TupleType, bool> cond, string reason) => new TypePattern(
        x => x switch
        {
            TupleType ttype => cond(ttype),
            _ => false,
        },
        "IsTuple" + (reason.Length == 0 ? string.Empty : $"&& {reason}"));

    /// <summary>
    /// <see cref="IsTuple(Func{TupleType, bool}, string)"/>.
    /// </summary>
    /// <returns>TypePattern.</returns>
    public static TypePattern IsTuple() => IsTuple(t => true, string.Empty);

    /// <summary>
    /// is void tuple type.
    /// </summary>
    public static TypePattern IsUnit() => IsTuple(t => t.Count == 0, "IsUnit");

    /// <summary>
    /// Check the datatype is None type.
    /// </summary>
    public static TypePattern IsNoneType() => new TypePattern(
        x => x switch
        {
            NoneType ntype => true,
            _ => false,
        },
        "IsNone");

    /// <summary>
    /// is scalar quant param.
    /// </summary>
    public static TypePattern IsQuantParamType() => IsScalar() & HasDataType(new QuantParamType());

    /// <summary>
    /// get padding windows output size.
    /// </summary>
    public static int GetWindowedOutputSize(int size, int filter, int stride, int dilation, bool same, bool ceilMode = false)
    {
        var effective_filter_size = ((filter - 1) * dilation) + 1;
        if (same)
        {
            return (size + stride - 1) / stride;
        }
        else
        {
            if (!ceilMode)
            {
                return (size - effective_filter_size + stride) / stride;
            }
            else
            {
                return (int)System.Math.Ceiling((float)(size - effective_filter_size + stride) / stride);
            }
        }
    }

    /// <summary>
    /// GetWindowedOutputSize.
    /// </summary>
    public static int GetWindowedOutputSize(int size, int filter, int stride, int dilation, (int Before, int After) padding)
    {
        var effective_filter_size = ((filter - 1) * dilation) + 1;
        return (size + padding.Before + padding.After - effective_filter_size + stride) / stride;
    }
}
