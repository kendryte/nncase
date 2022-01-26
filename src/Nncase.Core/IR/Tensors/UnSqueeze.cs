// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static Nncase.IR.TypePatternUtility;

namespace Nncase.IR.Tensors
{
    public sealed record UnSqueeze() : Op
    {
        public static ParameterInfo Input = new(typeof(UnSqueeze), 0, "input");

        public static ParameterInfo Dim = new(typeof(UnSqueeze), 1, "dim", IsRank(1) & IsIntegral());

        /// <inheritdoc/>
        public IRType InferInvokeResultType(ITypeInferenceContext context, TensorType input, TensorType dim)
        {
            if (context.GetArgument(this, Dim) is Const tdims)
            {
                var dimsValue = tdims.ToTensor<int>();
                var outShape = input.Shape.ToList();
                foreach (var dimVal in dimsValue)
                {
                    var dimV = Util.PositiveIndex(dimVal, input);
                    if (dimV < 0)
                    {
                        for (int i = dimV; i < 0; i++)
                        {
                            outShape.Insert(0, 1);                            
                        }
                    }
                }
                return input with { Shape = new Shape(outShape) };
            }
            return input with { Shape = new Shape(Enumerable.Repeat(Dimension.Unknown, input.Shape.Rank + 1)) };
        }
    }
}
