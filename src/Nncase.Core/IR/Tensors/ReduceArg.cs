// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static Nncase.IR.TypePatternUtility;

namespace Nncase.IR.Tensors
{
    /// <summary>
    /// ReduceArg expression.
    /// </summary>
    public sealed record ReduceArg(ReduceArgOp ReduceArgOp) : Op
    {
        /// <summary>
        /// Gets input.
        /// </summary>
        public static readonly ParameterInfo Input = new(typeof(ReduceArg), 0, "input");

        /// <summary>
        /// Gets Axis.
        /// </summary>
        // In torch named dim
        public static readonly ParameterInfo Axis = new(typeof(ReduceArg), 1, "axis", IsIntegralScalar());

        /// <summary>
        /// Gets KeepDims.
        /// </summary>
        public static readonly ParameterInfo KeepDims = new(typeof(ReduceArg), 2, "keepDims", IsBoolScalar());

        /// <summary>
        /// Gets depth.
        /// </summary>
        // only in onnx
        public static readonly ParameterInfo SelectLastIndex = new(typeof(ReduceArg), 3, "selectLastIndex", IsBoolScalar());

        /// <inheritdoc/>
        public IRType InferInvokeResultType(ITypeInferenceContext context, TensorType input, TensorType axis, TensorType keepDims, TensorType selectLastIndex)
        {
            if (context.GetArgument(this, Axis) is Const axisValue &&
                context.GetArgument(this, KeepDims) is Const keepDimsValue)
            {
                var shape = input.Shape.ToList();
                var axisIndex = axisValue.ToScalar<int>();
                axisIndex = axisIndex >= 0 ? axisIndex : input.Shape.Rank + axisIndex;
                if (keepDimsValue.ToScalar<bool>())
                {
                    shape[axisIndex] = 1;
                }
                else
                {
                    shape.RemoveAt(axisIndex);
                }
                return input with { Shape = new Shape(shape) };
            }
            else
            {
                return new InvalidType("ReduceArg axis and keepDims are not const");
            }
        }
    }
}
