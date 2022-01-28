// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
using System;
using Nncase.IR;

namespace Nncase.IR.Tensors
{
    public sealed record DeQuantize(DataType TargetType) : Op
    {
        public static ParameterInfo Input = new(typeof(DeQuantize), 0, "input");

        public static ParameterInfo ZeroPoint = new(typeof(DeQuantize), 1, "zero_point");

        public static ParameterInfo Scale = new(typeof(DeQuantize), 2, "scale");

        /// <inheritdoc/>
        public IRType InferInvokeResultType(ITypeInferenceContext context, TensorType input, TensorType zero_point, TensorType scale)
        {
            return new TensorType(TargetType, input.Shape);
        }
    }
}