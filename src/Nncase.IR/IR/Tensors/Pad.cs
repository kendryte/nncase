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
    public sealed record Pad(PadMode padMode) : Op
    {
        public static ParameterInfo Input = new(typeof(Pad), 0, "Input");
        public static ParameterInfo Pads = new(typeof(Pad), 1, "Pads");
        public static ParameterInfo Value = new(typeof(Pad), 2, "Value");

        /// <inheritdoc/>
        public override IRType InferInvokeResultType(ITypeInferenceContext context)
        {
            throw new NotImplementedException();
        }
    }
}
