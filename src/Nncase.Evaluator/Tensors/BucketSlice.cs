// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Linq;
using DryIoc.ImTools;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.NN;
using Nncase.IR.Tensors;
using OrtKISharp;
using static Nncase.IR.F.Tensors;

namespace Nncase.Evaluator.Tensors;

/// <summary>
/// Evaluator for <see cref="BucketSlice"/>.
/// </summary>
public class BucketSliceEvaluator : IEvaluator<BucketSlice>, ITypeInferencer<BucketSlice>, ICostEvaluator<BucketSlice>, IShapeEvaluator<BucketSlice>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, BucketSlice BucketSlice)
    {
        var input = context.GetArgumentValueAsTensor(BucketSlice, BucketSlice.Input);
        var originInput = context.GetArgumentValueAsTensor(BucketSlice, BucketSlice.OriginInput);
        var shape = originInput.Shape.ToValueArray();
        var rank = shape.Length;
        var begins = Enumerable.Repeat(0, rank).ToArray();
        return Slice(input, begins, shape, rank).Evaluate();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, BucketSlice target)
    {
        var input = context.CheckArgumentType<TensorType>(target, BucketSlice.Input);
        return Visit(context, target, input);
    }

    public Cost Visit(ICostEvaluateContext context, BucketSlice target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, BucketSlice.Input);
        var outputType = context.GetReturnType<TensorType>();
        return new Cost()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(inputType),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(outputType),
            [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(inputType, CostUtility.GetCPUCyclesOfBinary(BinaryOp.Mul)),
        };
    }

    public Expr Visit(IShapeEvaluateContext context, BucketSlice target) => 1;

    private IRType Visit(ITypeInferenceContext context, BucketSlice target, TensorType input)
    {
        var shape = context.GetArgument(target, BucketSlice.OriginInput);
        return shape.CheckedType;
    }
}
