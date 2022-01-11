// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NetFabric.Hyperlinq;

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
            if (shapeValue is Const constShapeValue && input.Shape.IsFixed)
            {
                return TypeInference.BroadcastType(input, new TensorType(input.DType, constShapeValue.ToArray<int>()));
            }
            else
            {
                return new InvalidType("Broadcast shape is unknown");
            }
        }
    }
}
