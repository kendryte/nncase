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
    public sealed record Squeeze() : Op
    {
        public static ParameterInfo Input = new(typeof(Squeeze), 0, "input");

        public static ParameterInfo Dim = new(typeof(Squeeze), 1, "dim", IsScalar(IsIntegral()));

        /// <inheritdoc/>
        public IRType InferInvokeResultType(ITypeInferenceContext context, TensorType input, TensorType dim)
        {
            if (!Dim.CheckType(dim))
                return new InvalidType("The Perm RanK Must Equal 1");
            if (context.GetArgument(this, Dim) is Const dim_con)
            {
                var dim_v = dim_con.ToScalar<int>();
                var outshape = input.Shape.ToList();
                if (outshape[dim_v].IsFixed && outshape[dim_v].FixedValue == 1)
                {
                    outshape.RemoveAt(dim_v);
                    return input with { Shape = new Shape(outshape) };
                }
                return new InvalidType("The Shape[dim] is not 1!");
            }
            return input with { Shape = new Shape(Enumerable.Repeat(Dimension.Unknown, input.Shape.Count() - 1)) };
        }
    }
}