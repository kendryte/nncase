// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

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
            var permExpr = context.GetArgument(this, Perm);
            return TypeInference.TransposeType(input, permExpr);
        }
    }
}
