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
    /// OneHot expression.
    /// </summary>
    public sealed record OneHot(OneHotMode OneHotMode) : Op
    {
        /// <summary>
        /// Gets input.
        /// </summary>
        public static readonly ParameterInfo Indices = new(typeof(OneHot), 0, "indices");

        /// <summary>
        /// Gets depth.
        /// </summary>
        public static readonly ParameterInfo Depth = new(typeof(OneHot), 1, "depth");

        /// <summary>
        /// Gets on_value.
        /// </summary>
        public static readonly ParameterInfo OnValue = new(typeof(OneHot), 2, "on_value");
        
        /// <summary>
        /// Gets off_value.
        /// </summary>
        public static readonly ParameterInfo OffValue = new(typeof(OneHot), 3, "off_value");

        /// <summary>
        /// Gets axis.
        /// </summary>
        public static readonly ParameterInfo Axis = new(typeof(OneHot), 4, "axis");
        
        /// <inheritdoc/>
        public IRType InferInvokeResultType(ITypeInferenceContext context, TensorType indices, TensorType depth, TensorType on_value, TensorType off_value, TensorType axis)
        {
            if (!Depth.CheckType(depth))
            {
                return new InvalidType("OneHot depth must be scalar");
            }
            
            if (!OnValue.CheckType(on_value))
            {
                return new InvalidType("OneHot on_value must be scalar");
            }
            
            if (!OffValue.CheckType(off_value))
            {
                return new InvalidType("OneHot off_value must be scalar");
            }
            
            if (!Axis.CheckType(axis))
            {
                return new InvalidType("OneHot axis must be scalar");
            }
            
            // indices_shape[:axis] + [depth] + indices_shape[axis:]
            if (context.GetArgument(this, Axis) is Const axisValue
                && context.GetArgument(this, Depth) is Const depthValue)
            {
                var newShape = indices.Shape.InsertAndClone(axisValue.ToScalar<int>(), depthValue.ToScalar<int>());
                return new TensorType(on_value.DType, newShape);
            }

            return new InvalidType("OneHot axis or depth is not const");
        }
    }
}
