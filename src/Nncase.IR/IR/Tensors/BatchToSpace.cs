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
    public sealed record BatchToSpace() : Op
    {
        public static readonly ParameterInfo Input = new(typeof(BatchToSpace), 0, "Input");

        public static readonly ParameterInfo BlockShape = new(typeof(BatchToSpace), 1, "BlockShape");

        public static readonly ParameterInfo Crops = new(typeof(BatchToSpace), 2, "Crops");

        public override IRType InferInvokeResultType(ITypeInferenceContext context)
        {
            throw new NotImplementedException();
        }
    }
}
