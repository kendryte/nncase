// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static Nncase.IR.Utility;
using Nncase;

namespace Nncase.IR.Tensors
{
    /// <summary>
    /// Reshape expression.
    /// </summary>
    public sealed record Reshape() : Op
    {
        /// <summary>
        /// Gets input.
        /// </summary>
        public static readonly ParameterInfo Input = new(typeof(Reshape), 0, "input");

        /// <summary>
        /// Gets shape.
        /// </summary>
        public static readonly ParameterInfo Shape = new(typeof(Reshape), 1, "shape", HasRank(1));

        /// <inheritdoc/>
        public IRType InferInvokeResultType(ITypeInferenceContext context, TensorType input, TensorType shape)
        {
            Shape.CheckTypeThrow(shape);
            if (context.GetArgument(this, Shape) is Const shape_con)
            {
                return input with { Shape = new IR.Shape(shape_con.ToTensor<int>()) };
            }
            return input with { Shape = IR.Shape.Unranked };
        }
    }
}
