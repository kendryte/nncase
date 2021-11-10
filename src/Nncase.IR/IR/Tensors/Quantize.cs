// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
using System;
using Nncase.IR;

namespace Nncase.IR.Tensors
{

    public sealed record Quantize(DataType TargetType) : Op
    {
        public static ParameterInfo Input = new(typeof(Quantize), 0, "input");

        public static ParameterInfo ZeroPoint = new(typeof(Quantize), 1, "zeroPoint");

        public static ParameterInfo Scale = new(typeof(Quantize), 2, "scale");

        /// <inheritdoc/>
        public IRType InferInvokeResultType(ITypeInferenceContext context)
        {
            throw new NotImplementedException();
        }
    }
}