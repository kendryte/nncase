// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Runtime.CompilerServices;

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

        public bool MatchLeaf(IRType ValueType) => Cond(ValueType);

        /// <summary>
        /// Check the irtype, if not equal, throw exception.
        /// </summary>
        /// <param name="ValueType">give the ir type.</param>
        /// <param name="FieldName"> the argument name.</param>
        /// <exception cref="InvalidOperationException"></exception>
        public T Check<T>(T ValueType, string FieldName) where T : IRType
        {
            if (ValueType is TensorType tensorValueType && tensorValueType.Shape.IsUnranked)
            {
                return ValueType;
            }
            if (ValueType == null || !MatchLeaf(ValueType))
            {
                var cur = ValueType is null ? "None" : CompilerServices.Print(ValueType);
                throw new InvalidOperationException($"{FieldName} Requrie <{Reason}>, But {cur}!");
            }

            return ValueType;
        }

        public static TypePattern operator &(TypePattern lhs, TypePattern rhs) => new TypePattern(x => lhs.Cond(x) && rhs.Cond(x), $"<{lhs.Reason}> And <{rhs.Reason}>");

        public static TypePattern operator |(TypePattern lhs, TypePattern rhs) => new TypePattern(x => lhs.Cond(x) || rhs.Cond(x), $"<{lhs.Reason}> Or <{rhs.Reason}>");

        public static TypePattern operator !(TypePattern lhs) => new TypePattern(x => !lhs.Cond(x), $"<Not {lhs.Reason}>");
    }

    public static partial class TypePatternUtility
    {
        /// <summary>
        /// the datatype is AnyType
        /// <remarks>
        /// NOTE it's mean the tensorType/TupleType is error
        /// </remarks>
        /// </summary>
        /// <returns></returns>
        public static TypePattern IsAnyType() => new TypePattern(AnyType.Default);

        /// <summary>
        /// custom the type cond
        /// </summary>
        /// <param name="TypeCond"></param>
        /// <param name="reason"></param>
        /// <returns></returns>
        public static TypePattern IsType(Func<IRType, bool> TypeCond, [CallerArgumentExpression("TypeCond")] string reason = null!) => new TypePattern(TypeCond, reason);

        /// <summary>
        /// this type is equal to target IRType
        /// </summary>
        /// <param name="Type"></param>
        /// <returns></returns>
        public static TypePattern IsType(IRType Type) => IsType(x => x == Type, $"Type = {Type.ToString()}");

        /// <summary>
        /// is IsIRType
        /// </summary>
        /// <returns></returns>
        public static TypePattern IsIRType() => IsType(t => true, "IsIRType");

        /// <summary>
        /// is shape
        /// </summary>
        /// <param name="shapeCond"></param>
        /// <param name="reason"></param>
        /// <returns></returns>
        public static TypePattern HasShape(Func<Shape, bool> shapeCond, string reason) => new TypePattern(x => x switch
        {

            TensorType ttype => shapeCond(ttype.Shape),
            _ => false,
        }, reason);

        /// <summary>
        /// the tensor has FixedShape
        /// </summary>
        /// <returns>TypePattern.</returns>
        public static TypePattern HasFixedShape() => HasShape(shape => shape.IsFixed, "HasFixedShape");

        /// <summary>
        /// the tensor has FixedShape
        /// </summary>
        /// <returns>TypePattern.</returns>
        public static TypePattern HasRank() => HasShape(shape => shape.IsRanked, "HasRank");

        /// <summary>
        /// is target shape
        /// </summary>
        /// <param name="target_shape"></param>
        /// <returns></returns>
        public static TypePattern HasShape(Shape target_shape) => HasShape(
          inshape =>
            inshape.Rank == target_shape.Rank &&
            inshape.Zip(target_shape).All(
              (dim) => dim.Item2 == Dimension.Unknown ? true : dim.Item2 == dim.Item1
            ), $"Shape = {target_shape.ToString()}");


        /// <summary>
        /// is custom rank
        /// </summary>
        /// <param name="cond"></param>
        /// <param name="reason"></param>
        /// <returns></returns>
        public static TypePattern HasRank(Func<int, bool> cond, string reason) => HasShape(
          inshape => cond(inshape.Rank), reason);

        /// <summary>
        /// is target rank
        /// </summary>
        /// <param name="rank"></param>
        /// <returns></returns>
        public static TypePattern HasRank(int rank) => HasRank(r => r == rank, $"Rank = {rank}");

        /// <summary>
        /// is tensor
        /// </summary>
        /// <returns></returns>
        public static TypePattern IsTensor() => new TypePattern(
          x => x switch
          {
              TensorType ttype => ttype.IsTensor,
              _ => false,
          }, "IsTensor"
        );

        /// <summary>
        /// check the tensor is handle
        /// </summary>
        /// <returns></returns>
        public static TypePattern IsPointer() => new TypePattern(
          x => (x is TensorType { IsScalar: true, DType: PointerType { ElemType: PrimType { } } }), "IsPointer"
        );

        /// <summary>
        /// check the datatype is scalar
        /// </summary>
        /// <returns></returns>
        public static TypePattern IsScalar() => new TypePattern(
          x => x switch
          {
              TensorType ttype => ttype.IsScalar,
              _ => false,
          },
          "IsScalar"
        );

        /// <summary>
        /// check the datatype is integral
        /// </summary>
        /// <param name="dataType"></param>
        /// <returns></returns>
        public static TypePattern HasDataType(DataType dataType) => new TypePattern(
          x => x switch
          {
              TensorType ttype => ttype.DType == dataType,
              _ => false
          }, $"{dataType.GetDisplayName()}"
        );

        /// <summary>
        /// check the data type IsIntegral
        /// </summary>
        /// <returns></returns>
        public static TypePattern IsIntegral() => new TypePattern(
          x => x switch
          {
              TensorType ttype => DataTypes.IsIntegral(ttype.DType),
              _ => false,
          }, "IsIntegral"
        );

        /// <summary>
        /// chenck the data type isfloat
        /// </summary>
        /// <returns></returns>
        public static TypePattern IsFloat() => new TypePattern(
          x => x switch
          {
              TensorType ttype => DataTypes.IsFloat(ttype.DType),
              _ => false,
          }, "IsFloat"
        );

        /// <summary>
        /// check the data type is bool
        /// </summary>
        /// <returns></returns>
        public static TypePattern IsBool() => new TypePattern(
            x => x switch
            {
                TensorType ttype => ttype.DType == DataTypes.Boolean,
                _ => false,
            }, "IsBool"
        );

        /// <summary>
        /// int scalar
        /// </summary>
        /// <returns></returns>
        public static TypePattern IsIntegralScalar() => IsScalar() & IsIntegral();

        /// <summary>
        /// bool scalar
        /// </summary>
        /// <returns></returns>
        public static TypePattern IsBoolScalar() => IsScalar() & IsBool();

        /// <summary>
        /// float scalar
        /// </summary>
        /// <returns></returns>
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
                _ => false
            }, "IsTuple" + (reason.Length == 0 ? "" : $"&& {reason}")
        );

        /// <summary>
        /// <see cref="IsTuple(Func{TupleType, bool}, string)"/>
        /// </summary>
        /// <returns>TypePattern.</returns>
        public static TypePattern IsTuple() => IsTuple(t => true, string.Empty);

        /// <summary>
        /// is void tuple type
        /// </summary>
        /// <returns></returns>
        public static TypePattern IsUnit() => IsTuple(t => t.Count == 0, "IsUnit");

        /// <summary>
        /// Check the datatype is None type.
        /// </summary>
        /// <returns></returns>
        public static TypePattern IsNoneType() => new TypePattern(x => x switch
        {
            NoneType ntype => true,
            _ => false
        }, "IsNone");

        /// <summary>
        /// is scalar quant param
        /// </summary>
        /// <returns></returns>
        public static TypePattern IsQuantParamType() => IsScalar() & HasDataType(new QuantParamType());

        /// <summary>
        /// get padding windows output size
        /// </summary>
        /// <param name="size"></param>
        /// <param name="filter"></param>
        /// <param name="stride"></param>
        /// <param name="dilation"></param>
        /// <param name="same"></param>
        /// <param name="ceilMode"></param>
        /// <returns></returns>
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
                    return (int)System.Math.Ceiling(((float)(size - effective_filter_size + stride) / stride));
                }
            }
        }

        /// <summary>
        /// GetWindowedOutputSize
        /// </summary>
        /// <param name="size"></param>
        /// <param name="filter"></param>
        /// <param name="stride"></param>
        /// <param name="dilation"></param>
        /// <param name="padding"></param>
        /// <returns></returns>
        public static int GetWindowedOutputSize(int size, int filter, int stride, int dilation, (int before, int after) padding)
        {
            var effective_filter_size = (filter - 1) * dilation + 1;
            return (size + padding.before + padding.after - effective_filter_size + stride) / stride;
        }
    }
}
