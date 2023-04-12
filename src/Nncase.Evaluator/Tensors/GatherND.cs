// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Linq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Tensors;
using OrtKISharp;

namespace Nncase.Evaluator.Tensors;

/// <summary>
/// Evaluator for <see cref="GatherND"/>.
/// </summary>
public class GatherNDEvaluator : IEvaluator<GatherND>, ITypeInferencer<GatherND>, ICostEvaluator<GatherND>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, GatherND gatherND)
    {
        var input = context.GetOrtArgumentValue(gatherND, GatherND.Input);
        var indices = context.GetInt64OrtTensorArgumentValue(gatherND, GatherND.Index);
        var batchDims = context.GetArgumentValueAsScalar<long>(gatherND, GatherND.BatchDims);
        return OrtKI.GatherND(input, indices, batchDims).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, GatherND target)
    {
        var input = context.CheckArgumentType<TensorType>(target, GatherND.Input);
        var batchDims = context.CheckArgumentType<TensorType>(target, GatherND.BatchDims);
        var index = context.CheckArgumentType<TensorType>(target, GatherND.Index);
        return Visit(context, target, input, batchDims, index);
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, GatherND target)
    {
        var returnType = context.GetReturnType<IRType>();

        return new()
        {
            [CostFactorNames.MemoryLoad] = returnType switch
            {
                TensorType t => CostUtility.GetMemoryAccess(t),
                _ => 1,
            },
            [CostFactorNames.MemoryStore] = returnType switch
            {
                TensorType t => CostUtility.GetMemoryAccess(t),
                _ => 1,
            },
        };
    }

    private IRType Visit(ITypeInferenceContext context, GatherND target, TensorType input, TensorType batchDims, TensorType index)
    {
        if (context.GetArgument(target, GatherND.BatchDims) is TensorConst batchDimsValue)
        {
            var lastIndexDims = index.Shape[index.Shape.Count - 1];
            if (!lastIndexDims.IsFixed)
            {
                return new InvalidType("GatherND input last dim is dynamic, can't infer result shape");
            }

            // result shape = index_shape[:-1] + input_shape[index_shape[-1] + batch_dims:]
            var dimensions = index.Shape.ToArray()[..(index.Shape.Rank - 1)];
            var d = lastIndexDims.FixedValue + batchDimsValue.Value.ToScalar<int>();
            var shapeValue = dimensions.Concat(input.Shape.ToArray()[d..]);
            return new TensorType(input.DType, new IR.Shape(shapeValue));
        }
        else
        {
            return new InvalidType("GatherND batch_dims must be constant");
        }
    }
}
