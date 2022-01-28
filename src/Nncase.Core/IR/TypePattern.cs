// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;

namespace Nncase.IR
{
    public record TypePattern(Func<IRType, bool> Cond, string Reason)
    {
        public TypePattern(IRType ValueType) : this(x => (x == ValueType), $"Type = {ValueType.ToString()}") { }
        public TypePattern(AnyType ValueType) : this(x => (x == ValueType), $"Type = {ValueType.ToString()}") { }
        public TypePattern(TensorType ValueType) : this(x => (x == ValueType), $"Type = {ValueType.ToString()}") { }
        public TypePattern(InvalidType ValueType) : this(x => (x == ValueType), $"Type = {ValueType.ToString()}") { }
        public TypePattern(TupleType ValueType) : this(x => (x == ValueType), $"Type = {ValueType.ToString()}") { }
        public TypePattern(CallableType ValueType) : this(x => (x == ValueType), $"Type = {ValueType.ToString()}") { }

        public bool MatchLeaf(IRType? ValueType) => ValueType is not null ? Cond(ValueType) : false;

        /// <summary>
        /// Check the irtype, if not equal, throw exception.
        /// </summary>
        /// <param name="ValueType"></param>
        /// <exception cref="InvalidOperationException"></exception>
        public T Check<T>(T? ValueType) where T : IRType
        {
            if (ValueType == null || !MatchLeaf(ValueType))
                throw new InvalidOperationException($"Requrie <{Reason}>, But {ValueType}!");
            return ValueType;
        }

        public static TypePattern operator &(TypePattern lhs, TypePattern rhs) => new TypePattern(x => lhs.Cond(x) && rhs.Cond(x), $"<{lhs.Reason}> And <{rhs.Reason}>");

        public static TypePattern operator |(TypePattern lhs, TypePattern rhs) => new TypePattern(x => lhs.Cond(x) || rhs.Cond(x), $"<{lhs.Reason}> Or <{rhs.Reason}>");
    }

    public static partial class Utility
    {
        public static TypePattern IsAnyType() => new TypePattern(AnyType.Default);

        public static TypePattern HasType(Func<IRType, bool> TypeCond, string reason) => new TypePattern(TypeCond, reason);

        public static TypePattern IsIRType() => HasType(t => true, "IsIRType");

        public static TypePattern HasType(IRType Type) => HasType(x => x == Type, $"Type = {Type.ToString()}");

        public static TypePattern HasDType(Func<DataType, bool> DTypeCond, string reason) => new TypePattern(x => x switch
         {
             TensorType ttype => DTypeCond(ttype.DType),
             _ => false,
         }, reason);

        public static TypePattern HasDType(DataType DType) => HasDType((DataType x) => x == DType, $"DType = {DType.ToString()}");

        public static TypePattern HasShape(Func<Shape, bool> shapeCond, string reason) => new TypePattern(x => x switch
             {

                 TensorType ttype => ttype.IsTensor && shapeCond(ttype.Shape),
                 _ => false,
             }, reason);

        public static TypePattern HasShape(Shape target_shape) => HasShape(
          inshape =>
            inshape.Rank == target_shape.Rank &&
            inshape.Zip(target_shape).All(
              (dim) => dim.Item2 == Dimension.Unknown ? true : dim.Item2 == dim.Item1
            ), $"Shape = {target_shape.ToString()}");

        public static TypePattern HasRank(Func<int, bool> cond, string reason) => HasShape(
          inshape => cond(inshape.Rank), reason);

        public static TypePattern HasRank(int rank) => HasRank(r => r == rank, $"Rank = {rank}");

        public static TypePattern IsTensor() => new TypePattern(
          x => x switch
          {
              TensorType ttype => ttype.IsTensor,
              _ => false,
          }, "IsTensor"
        );

        /// <summary>
        /// The void unit.
        /// </summary>
        /// <returns></returns>
        public static TypePattern IsUnit() => new TypePattern(
          x => x == TupleType.Void, "IsUnit"
        );

        /// <summary>
        ///
        /// </summary>
        /// <returns></returns>
        public static TypePattern IsHandle() => new TypePattern(
          x => x is HandleType, "IsHandle"
        );

        public static TypePattern IsScalar() => new TypePattern(
          x => x switch
          {
              TensorType ttype => ttype.IsScalar,
              _ => false,
          },
          "IsScalar"
        );

        public static TypePattern IsIntegral(DataType dataType) => new TypePattern(
          x => x switch
          {
              TensorType ttype => ttype.DType == dataType,
              _ => false,
          }, $"IsIntegral {dataType}"
        );

        public static TypePattern IsIntegral() => new TypePattern(
          x => x switch
          {
              TensorType ttype => DataTypes.IsIntegral(ttype.DType),
              _ => false,
          }, "IsIntegral"
        );

        public static TypePattern IsFloat(DataType dataType) => new TypePattern(
          x => x switch
          {
              TensorType ttype => ttype.DType == dataType,
              _ => false,
          }, $"IsFloat {dataType}"
        );

        public static TypePattern IsFloat() => new TypePattern(
          x => x switch
          {
              TensorType ttype => DataTypes.IsFloat(ttype.DType),
              _ => false,
          }, "IsFloat"
        );

        public static TypePattern IsBool() => new TypePattern(
            x => x switch
            {
                TensorType ttype => ttype.DType == DataType.Bool,
                _ => false,
            }, "IsBool"
        );

        public static TypePattern IsIntegralScalar() => IsScalar() & IsIntegral();

        public static TypePattern IsBoolScalar() => IsScalar() & IsIntegral();

        public static TypePattern IsFloatScalar() => IsScalar() & IsFloat();

        public static int GetWindowedOutputSize(int size, int filter, int stride, int dilation, bool same, bool ceilMode = false)
        {
            var effective_filter_size = (filter - 1) * dilation + 1;
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
                    return (int)System.Math.Ceiling(((float)(size - effective_filter_size + stride) / stride));
                }
            }
        }
    }
}