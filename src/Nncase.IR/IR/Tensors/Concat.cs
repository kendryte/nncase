// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.IR.Tensors
{
    /// <summary>
    /// Concat expression.
    /// </summary>
    public sealed record Concat() : Op
    {
        /// <summary>
        /// Gets input.
        /// </summary>
        public static readonly ParameterInfo Input = new(typeof(Concat), 0, "inputs");

        /// <summary>
        /// Gets axis.
        /// </summary>
        public static readonly ParameterInfo Axis = new(typeof(Concat), 1, "axis");


        // axis: if one of inputs shape[axis] is unknown
        // then dims axis is known
        // else get sum of dims
        private Dimension AxisDim(TupleType inputs, int axisValue)
        {
            var allAxisDimIsFixed = inputs.Fields.Aggregate(
                true,
                (prod, next) => prod && (next as TensorType).Shape[axisValue].IsFixed);
            if (allAxisDimIsFixed)
            {
                return inputs.Fields.Aggregate(
                    0,
                    (prod, next) => prod + (next as TensorType).Shape[axisValue].FixedValue);
            }
            else
            {
                return Dimension.Unknown;
            }
        }
        
        /// <inheritdoc/>
        public IRType InferInvokeResultType(ITypeInferenceContext context, TupleType inputs, TensorType axis)
        {
            var input0 = inputs[0] as TensorType;
            InvalidType? invalidType = null;
            var axisValue = (context.GetArgument(this, Axis) as Const).ToScalar<int>();
            var shapeValue = Enumerable.Range(0, input0.Shape.Rank).Select(i =>
            {
                if (i == axisValue)
                {
                    return AxisDim(inputs, axisValue);
                }
                // if all input shape[dim] is not same, return invalid
                else
                {
                    var allAxisDimIsSame = inputs.Fields.Aggregate(
                        true,
                        (prod, next) => prod && (next as TensorType).Shape[i].IsFixed);
                    if (allAxisDimIsSame)
                    {
                        return (inputs[0] as TensorType).Shape[0];
                    }
                    else
                    {
                        invalidType = new InvalidType("Concat dims that except the shape of axis dim are different");
                        return Dimension.Unknown;
                    }
                }
            });
            var shape = new Shape(shapeValue);
            return (IRType) invalidType?? new TensorType(input0.DType, shape);
        }
    }
}
