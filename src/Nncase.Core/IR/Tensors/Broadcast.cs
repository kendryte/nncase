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
    /// Broadcast expression.
    /// </summary>
    public sealed record Broadcast() : Op
    {
        /// <summary>
        /// Gets input.
        /// </summary>
        public static readonly ParameterInfo Input = new(typeof(Broadcast), 0, "input");

        /// <summary>
        /// Gets input.
        /// </summary>
        public static readonly ParameterInfo Shape = new(typeof(Broadcast), 1, "shape");

        /// <inheritdoc/>
        public IRType InferInvokeResultType(ITypeInferenceContext context, TensorType input, TensorType shape)
        {
            var shapeValue = context.GetArgument(this, Shape);
            if (shapeValue is Const && input.Shape.IsFixed)
            {
                for (var i = input.Shape.Rank; i > 0; --i)
                {
                    var shapeShapeValue = shape.Shape[shape.Shape.Rank - i].FixedValue;
                    var inputShapeValue = input.Shape[input.Shape.Rank - i].FixedValue;
                    if (shapeShapeValue != inputShapeValue && inputShapeValue != 1)
                    {
                        return new InvalidType("Broadcast input shape and param shape is not match");
                    }
                }

                return shape;
            }
            else
            {
                return new InvalidType("Broadcast shape is unknown");
            }
        }
    }
}
