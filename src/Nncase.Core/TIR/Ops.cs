// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using static Nncase.IR.TypePatternUtility;

namespace Nncase.TIR
{
    /// <summary>
    /// <see cref="T.Load(Var, Expr)"/>
    /// </summary>
    public record Load() : Op
    {
        public static readonly ParameterInfo Handle = new(typeof(Load), 0, "handle", IsHandle());

        public static readonly ParameterInfo Index = new(typeof(Load), 1, "index", IsDataType(DataType.Int32) & (IsScalar() | IsRank(1)));

        /// <inheritdoc/>
        public IRType InferInvokeResultType(ITypeInferenceContext context, TensorType handle, TensorType index)
        {
            int lanes = index.IsScalar ? 1 : index.Shape[0].FixedValue;
            if (handle is TensorType { DType: PointerType { ElemType: PrimType etype } })
            {
                return new TensorType(etype with { Lanes = lanes }, Shape.Scalar);
            }
            return new InvalidType("Handle Is Not Valid!");
        }
    }

    /// <summary>
    /// <see cref="T.Ramp(Expr, Expr, int)"/>
    /// </summary>
    public record Ramp(int Lanes) : Op
    {
        public static readonly ParameterInfo Offset = new(typeof(Ramp), 0, "offset", IsDataType(DataType.Int32) & IsScalar());

        public static readonly ParameterInfo Stride = new(typeof(Ramp), 1, "stride", IsDataType(DataType.Int32) & IsScalar());

        /// <inheritdoc/>
        public IRType InferInvokeResultType(ITypeInferenceContext context, TensorType offset, TensorType stride)
        {
            // TODO maybe need simpify when the Lanes==1.
            return new TensorType(DataType.Int32, new Shape(Lanes));
        }
    }

    /// <summary>
    /// Store, return unit
    /// </summary>
    public sealed record Store() : Op
    {
        /// <summary>
        ///The buffer variable handle.
        /// </summary>
        public static readonly ParameterInfo Handle = new(typeof(Store), 0, "handle", IsHandle());

        /// <summary>
        ///The index locations to be stored.
        /// </summary>
        public static readonly ParameterInfo Index = new(typeof(Store), 1, "index", IsDataType(DataType.Int32));

        /// <summary>
        ///The value to be stored.
        /// </summary>
        public static readonly ParameterInfo Value = new(typeof(Store), 2, "value");

        /// <inheritdoc/>
        public IRType InferInvokeResultType(ITypeInferenceContext context, TensorType handle, TensorType index, TensorType value)
        {
            var lanes = index.IsScalar ? 1 : index.Shape[0].FixedValue;
            if (handle.PointedDType() != value.DType)
            {
                return new InvalidType($"You Can't Load The {value.DType} To {handle.DType}");
            }
            if (value.DType is PrimType ptype && ptype.Lanes != lanes)
            {
                return new InvalidType($"You're Index Lanes {lanes} Is Not Equal Value Lanes {ptype.Lanes}");
            }
            return TupleType.Void;
        }

    }
}