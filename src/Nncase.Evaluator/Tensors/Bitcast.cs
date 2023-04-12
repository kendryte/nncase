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
        var input = context.GetArgumentValue(cast, Bitcast.Input).AsTensor();
        var shape = context.GetArgumentValueAsArray<long>(cast, Bitcast.NewShape);
        return OrtKI.Reshape(input.CastTo(cast.NewType).ToOrtTensor(), shape, 0).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Bitcast target)
    {
        var input = context.CheckArgumentType<TensorType>(target, Bitcast.Input);
        return Visit(target, input);
    }

    /// <inheritdoc/>
    public string Visit(IIRPrinterContext context, Bitcast target, bool iLmode)
    {
        return $"{CompilerServices.Print(target.NewType)}({context.GetArgument(target, Cast.Input)})";
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, Bitcast target)
    {
        var input = context.GetArgumentType<TensorType>(target, Bitcast.Input);
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(input.DType),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(target.NewType),
            [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(target.NewType, 1),
        };
    }

    private IRType Visit(Bitcast target, TensorType input)
    {
        return new TensorType(target.NewType, input.Shape);
    }
}
