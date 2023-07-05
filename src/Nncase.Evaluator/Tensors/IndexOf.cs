// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Linq;
using DryIoc.ImTools;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Tensors;
using OrtKISharp;

namespace Nncase.Evaluator.Tensors;

/// <summary>
/// Evaluator for <see cref="IndexOf"/>.
/// </summary>
public class IndexOfEvaluator : IEvaluator<IndexOf>, ITypeInferencer<IndexOf>, ICostEvaluator<IndexOf>, IShapeEvaluator<IndexOf>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, IndexOf indexOf)
    {
        var input = context.GetArgumentValueAsArray<int>(indexOf, IndexOf.Input);
        var v = context.GetArgumentValueAsScalar<int>(indexOf, IndexOf.Value);
        return Value.FromTensor(input.IndexOf(v));
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, IndexOf target)
    {
        var input = context.CheckArgumentType<TensorType>(target, IndexOf.Input);
        return Visit(context, target, input);
    }

    public Cost Visit(ICostEvaluateContext context, IndexOf target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, IndexOf.Input);
        var outputType = context.GetReturnType<TensorType>();
        return new Cost()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(inputType),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(outputType),
            [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(inputType, CostUtility.GetCPUCyclesOfBinary(BinaryOp.Mul)),
        };
    }

    public Expr Visit(IShapeEvaluateContext context, IndexOf target) => System.Array.Empty<int>();

    private IRType Visit(ITypeInferenceContext context, IndexOf target, TensorType input)
    {
        return new TensorType(DataTypes.Int32, Shape.Scalar);
    }
}
