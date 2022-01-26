// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using static Nncase.IR.TypePatternUtility;

namespace Nncase.IR.Tensors
{
    /// <summary>
    /// Reshape expression.
    /// </summary>
    public sealed record Reduce(ReduceOp ReduceOp) : Op
    {
        public static readonly ParameterInfo Input = new(typeof(Reduce), 0, "input");
        public static readonly ParameterInfo Axis = new(typeof(Reduce), 1, "axis", IsIntegral() & IsRank(1));
        public static readonly ParameterInfo InitValue = new(typeof(Reduce), 2, "initValue", IsScalar());
        public static readonly ParameterInfo KeepDims = new(typeof(Reduce), 3, "keepDims", IsScalar() & IsIntegral());

        /// <inheritdoc/>
        public IRType InferInvokeResultType(ITypeInferenceContext context, TensorType input, TensorType axis,
          TensorType initValue, TensorType keepDims)
        {
            var args = context.GetArguments(this, KeepDims, Axis);
            return TypeInference.ReduceType(input, args[0], args[1]);
        }
    }
}
