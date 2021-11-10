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
    /// Reshape expression.
    /// </summary>
    public sealed record Reduce(ReduceOp ReduceOp) : Op
    {
        public static readonly ParameterInfo Input = new(typeof(Reduce), 0, "Input");
        public static readonly ParameterInfo Axis = new(typeof(Reduce), 1, "Axis");
        public static readonly ParameterInfo InitValue = new(typeof(Reduce), 2, "InitValue");
        public static readonly ParameterInfo KeepDims = new(typeof(Reduce), 3, "KeepDims");

        /// <inheritdoc/>
        public IRType InferInvokeResultType(ITypeInferenceContext context)
        {
            throw new NotImplementedException();
        }
    }
}
