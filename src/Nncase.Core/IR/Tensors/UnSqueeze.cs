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
    public sealed record UnSqueeze() : Op
    {
        public static ParameterInfo Input = new(typeof(UnSqueeze), 0, "input");

        public static ParameterInfo Dim = new(typeof(UnSqueeze), 1, "dim", IsScalar() & IsIntegral());

        /// <inheritdoc/>
        public IRType InferInvokeResultType(ITypeInferenceContext context, TensorType input, TensorType dim)
        {
            if (context.GetArgument(this, Dim) is Const tdims)
            {
                var dimv = tdims.ToScalar<int>();
                var outshape = input.Shape.ToList();
                if (dimv >= 0)
                    outshape.Insert(dimv, 1);
                else
                    outshape.Insert(outshape.Count + dimv, 1);
                return input with { Shape = new Shape(outshape) };
            }
            return input with { Shape = new Shape(Enumerable.Repeat(Dimension.Unknown, input.Shape.Rank + 1)) };
        }
    }
}
