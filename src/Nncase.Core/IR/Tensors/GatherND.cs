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
    /// GatherND expression.
    /// </summary>
    public sealed record GatherND() : Op
    {
        /// <summary>
        /// Gets input.
        /// </summary>
        public static readonly ParameterInfo Input = new(typeof(GatherND), 0, "input");

        /// <summary>
        /// Gets batch dims.
        /// </summary>
        public static readonly ParameterInfo BatchDims = new(typeof(GatherND), 1, "batch_dims");

        /// <summary>
        /// Gets index.
        /// </summary>
        public static readonly ParameterInfo Index = new(typeof(GatherND), 2, "index");

        /// <inheritdoc/>
        public IRType InferInvokeResultType(ITypeInferenceContext context, TensorType input, TensorType batch_dims, TensorType index)
        {
            if (context.GetArgument(this, BatchDims) is Const batchDimsValue)
            {
                var lastIndexDims = index.Shape.Last();
                if (!lastIndexDims.IsFixed)
                {
                    return new InvalidType("GatherND input last dim is dynamic, can't infer result shape");
                }

                // result shape = index_shape[:-1] + input_shape[index_shape[-1] + batch_dims:]
                var dimensions = index.Shape.ToArray()[..(index.Shape.Rank - 1)];
                var d = lastIndexDims.FixedValue + batchDimsValue.ToScalar<int>();
                var shapeValue = dimensions.Concat(input.Shape.ToArray()[d..]);
                return new TensorType(input.DType, new Shape(shapeValue));
            }
            else
            {
                return new InvalidType("GatherND batch_dims must be constant");
            }
        }
    }
}
