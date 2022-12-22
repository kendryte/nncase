// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Tensors;
using OrtKISharp;

namespace Nncase.Evaluator.Tensors;

/// <summary>
/// Evaluator for <see cref="Bitcast"/>.
/// </summary>
public class BitcastEvaluator : IEvaluator<Bitcast>, ITypeInferencer<Bitcast>, IOpPrinter<Bitcast>, ICostEvaluator<Bitcast>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, Bitcast cast)
    {
        var input = context.GetArgumentValue(cast, Cast.Input).AsTensor();
        return Value.FromTensor(input.CastTo(cast.newType));
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Bitcast target)
    {
        var input = context.CheckArgumentType<TensorType>(target, Cast.Input);
        return Visit(target, input);
    }

    /// <inheritdoc/>
    public string Visit(IIRPrinterContext context, Bitcast target, bool iLmode)
    {
        return $"{CompilerServices.Print(target.newType)}({context.GetArgument(target, Cast.Input)})";
    }

    /// <inheritdoc/>
    public Cost? Visit(ICostEvaluateContext context, Bitcast target)
    {
        var input = context.GetArgumentType<TensorType>(target, Bitcast.Input);
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(input.DType),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(target.newType),
            [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(target.newType, 1),
        };
    }

    private IRType Visit(Bitcast target, TensorType input)
    {
        return new TensorType(target.newType, input.Shape);
    }
}
