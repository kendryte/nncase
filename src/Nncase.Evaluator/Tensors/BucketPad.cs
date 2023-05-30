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
/// Evaluator for <see cref="BucketPad"/>.
/// </summary>
public class BucketPadEvaluator : IEvaluator<BucketPad>, ITypeInferencer<BucketPad>, ICostEvaluator<BucketPad>, IShapeEvaluator<BucketPad>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, BucketPad bucketPad)
    {
        var input = context.GetArgumentValueAsTensor(bucketPad, BucketPad.Input);
        var shape = context.GetArgumentValueAsArray<int>(bucketPad, BucketPad.Shape);
        var pads = shape - (Expr)input.Shape;
        var paddings = Transpose(
            Stack(new Tuple(Enumerable.Repeat(0, shape.Length).ToArray(), pads), 0),
            new[] { 1, 0 });
        var fixedInput = IR.F.NN.Pad(input, paddings, PadMode.Constant, Cast(0, input.ElementType)).Evaluate();
        return fixedInput;
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, BucketPad target)
    {
        var input = context.CheckArgumentType<TensorType>(target, BucketPad.Input);
        return Visit(context, target, input);
    }

    public Cost Visit(ICostEvaluateContext context, BucketPad target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, BucketPad.Input);
        var outputType = context.GetReturnType<TensorType>();
        return new Cost()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(inputType),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(outputType),
            [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(inputType, CostUtility.GetCPUCyclesOfBinary(BinaryOp.Mul)),
        };
    }

    public Expr Visit(IShapeEvaluateContext context, BucketPad target) => context.GetArgument(target, BucketPad.Shape);

    private IRType Visit(ITypeInferenceContext context, BucketPad target, TensorType input)
    {
        var shape = context.GetArgument(target, BucketPad.Shape);
        if (shape is TensorConst shapeConst)
        {
            return new TensorType(input.DType, shapeConst.Value.ToArray<int>());
        }

        return new InvalidType("BucketPad Shape need const");
    }
}
