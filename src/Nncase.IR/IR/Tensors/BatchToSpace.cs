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
        public static readonly ParameterInfo Input = new(typeof(BatchToSpace), 0, "input");

        public static readonly ParameterInfo BlockShape = new(typeof(BatchToSpace), 1, "blockShape");

        public static readonly ParameterInfo Crops = new(typeof(BatchToSpace), 2, "crops");

        public IRType InferInvokeResultType(ITypeInferenceContext context, TensorType input, TensorType blockShape, TensorType crops)
        {
            var newShape = input.Shape.ToList();
            newShape[0] = input.Shape[0] / blockShape.Shape.Prod();
            if (context.GetArgument(this, Crops) is Const cropsValue)
            {
                if (crops.Shape.Rank != 2)
                {
                    return new InvalidType("BatchToSpace crops rank must be 2");
                }

                var cropsV = cropsValue.ToTensor<int>();
                var afterCropShape = Enumerable.Range(0, crops.Shape.Rank).Select(
                    i => input.Shape[i + 1] * blockShape.Shape[0] - cropsV[i, 0] - cropsV[i, 1]);
                return new TensorType(input.DType, input.Shape.InsertAndClone(1, afterCropShape));
            }
            else
            {
                return new InvalidType("BatchToSpace can't infer shape with dynamic crops");
            }
        }
    }
}
