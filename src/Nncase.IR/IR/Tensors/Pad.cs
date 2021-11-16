// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Numerics.Tensors;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.IR.Tensors
{
    public sealed record Pad(PadMode PadMode) : Op
    {
        public static ParameterInfo Input = new(typeof(Pad), 0, "input");
        public static ParameterInfo Pads = new(typeof(Pad), 1, "pads");
        public static ParameterInfo Value = new(typeof(Pad), 2, "value");

        /// <inheritdoc/>
        public IRType InferInvokeResultType(ITypeInferenceContext context, TensorType input, TensorType pads, TensorType value)
        {
            Input.CheckTypeThrow(input);
            Pads.CheckTypeThrow(pads);
            Value.CheckTypeThrow(value);
            if (context.GetArgument(this, Pads) is Const paddings)
            {
                var (padH, padW) = Paddings.GetPaddingFromConst(paddings);
                var newShape = input.Shape.ToList();
                newShape[2] += padH.Sum;
                newShape[3] += padW.Sum;
                return new TensorType(input.DType, new Shape(newShape));
            }
            else
            {
                return new InvalidType("Pad paddings is dynamic, can't infer shape");
            }
        }
    }


}
