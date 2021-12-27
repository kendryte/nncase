// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static Nncase.IR.Utility;

namespace Nncase.IR.Tensors
{
    /// <summary>
    /// Transpose expression.
    /// </summary>
    public sealed record Transpose() : Op
    {
        /// <summary>
        /// Gets input.
        /// </summary>
        public static readonly ParameterInfo Input = new(typeof(Transpose), 0, "input");

        /// <summary>
        /// Gets perm.
        /// </summary>
        public static readonly ParameterInfo Perm = new(typeof(Transpose), 1, "perm", HasRank(1) & IsIntegral());

        /// <inheritdoc/>
        public IRType InferInvokeResultType(ITypeInferenceContext context, TensorType input, TensorType perm)
        {
            if (context.GetArgument(this, Perm) is Const perm_con)
            {
                if (input.Shape.IsUnranked)
                {
                    return new InvalidType("Transpose input should not be Unranked");
                }
                var permt = perm_con.ToTensor<int>();
                var inShape = input.Shape;
                var outShape = inShape.ToArray();
                foreach (var i in Enumerable.Range(0, inShape.Rank))
                {
                    outShape[i] = inShape[permt[i]];
                }
                return input with { Shape = outShape };
            }
            return input with { Shape = new Shape(Enumerable.Repeat(Dimension.Unknown, input.Shape.Rank)) };
        }
    }
}
