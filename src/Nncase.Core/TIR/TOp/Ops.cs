// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using static Nncase.IR.Utility;

namespace Nncase.TIR
{
    /// <summary>
    /// <see cref="F.TOp.Load(Var, Expr)"/>
    /// </summary>
    public record Load() : Op
    {
        public static readonly ParameterInfo Handle = new(typeof(Load), 0, "handle");

        public static readonly ParameterInfo Index = new(typeof(Load), 1, "index", IsIntegral(DataType.Int32) & (IsScalar() | HasRank(1)));

        /// <inheritdoc/>
        public IRType InferInvokeResultType(ITypeInferenceContext context, PointerType handle, TensorType index)
        {
            int lanes = index.IsScalar ? 1 : index.Shape[0].FixedValue;
            return new TensorType(handle.DType with { Lanes = lanes }, Shape.Scalar);
        }
    }

    /// <summary>
    /// <see cref="F.TOp.Ramp(Expr, Expr, int)"/>
    /// </summary>
    public record Ramp(int Lanes) : Op
    {
        public static readonly ParameterInfo Offset = new(typeof(Ramp), 0, "offset", IsIntegral(DataType.Int32) & IsScalar());

        public static readonly ParameterInfo Stride = new(typeof(Ramp), 1, "stride", IsIntegral(DataType.Int32) & IsScalar());

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
        public static readonly ParameterInfo Handle = new(typeof(Store), 0, "handle");
        /// <summary>
        ///The value to be stored.
        /// </summary>
        public static readonly ParameterInfo Value = new(typeof(Store), 1, "value");
        /// <summary>
        ///The index locations to be stored.
        /// </summary>
        public static readonly ParameterInfo Index = new(typeof(Store), 2, "index", IsIntegral(DataType.Int32));

        /// <inheritdoc/>
        public IRType InferInvokeResultType(ITypeInferenceContext context, PointerType handle, TensorType value, TensorType index)
        {
            var lanes = index.IsScalar ? 1 : index.Shape[0].FixedValue;
            if (handle.DType != value.DType)
            {
                return new InvalidType($"You Can't Load The {value.DType} To {handle.DType}");
            }
            if (value.DType.Lanes != lanes)
            {
                return new InvalidType($"You're Index Lanes {lanes} Is Not Equal Value Lanes {handle.DType.Lanes}");
            }
            return TupleType.Void;
        }


    }

}