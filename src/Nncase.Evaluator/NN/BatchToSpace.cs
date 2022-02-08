// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.NN;
using TorchSharp;
using torchF = TorchSharp.torch.nn.functional;

namespace Nncase.Evaluator.NN;

/// <summary>
/// Evaluator for <see cref="BatchToSpace"/>.
/// </summary>
public class BatchToSpaceEvaluator : IEvaluator<BatchToSpace>, ITypeInferencer<BatchToSpace>
{
    /// <inheritdoc/>
    public Const Visit(IEvaluateContext context, BatchToSpace target)
    {
        throw new NotImplementedException();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, BatchToSpace target)
    {
        var input = context.CheckArgumentType<TensorType>(target, BatchToSpace.Input);
        var blockShape = context.CheckArgumentType<TensorType>(target, BatchToSpace.BlockShape);
        var crops = context.CheckArgumentType<TensorType>(target, BatchToSpace.Crops);
        return Visit(context, target, input, blockShape, crops);
    }

    private IRType Visit(ITypeInferenceContext context, BatchToSpace target, TensorType input, TensorType blockShape, TensorType crops)
    {
        var newShape = input.Shape.ToList();
        newShape[0] = input.Shape[0] / blockShape.Shape.Prod();
        if (context.GetArgument(target, BatchToSpace.Crops) is Const cropsValue)
        {
            if (crops.Shape.Rank != 2)
            {
                return new InvalidType("BatchToSpace crops rank must be 2");
            }

            var cropsV = cropsValue.ToTensor<int>();
            var afterCropShape = Enumerable.Range(0, crops.Shape.Rank).Select(
                i => (input.Shape[i + 1] * blockShape.Shape[0]) - cropsV[i, 0] - cropsV[i, 1]);
            return new TensorType(input.DType, input.Shape.InsertAndClone(1, afterCropShape));
        }
        else
        {
            return new InvalidType("BatchToSpace can't infer shape with dynamic crops");
        }
    }
}
