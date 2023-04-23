// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Tensors;
using OrtKISharp;

namespace Nncase.Evaluator.Tensors;

/// <summary>
/// Evaluator for <see cref="Cast"/>.
/// </summary>
public class CastEvaluator : IEvaluator<Cast>, ITypeInferencer<Cast>, IOpPrinter<Cast>, ICostEvaluator<Cast>, IShapeEvaluator<Cast>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, Cast cast)
    {
        var input = context.GetArgumentValue(cast, Cast.Input).AsTensor();
        return Value.FromTensor(input.CastTo(cast.NewType, cast.CastMode));
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Cast target)
    {
        var input = context.CheckArgumentType<TensorType>(target, Cast.Input);
        return Visit(target, input);
    }

    /// <inheritdoc/>
    public string Visit(IIRPrinterContext context, Cast target, bool iLmode)
    {
        return $"{CompilerServices.Print(target.NewType)}({context.GetArgument(target, Cast.Input)})";
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, Cast target)
    {
        var input = context.GetArgumentType<TensorType>(target, Cast.Input);
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(input.DType),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(target.NewType),
            [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(target.NewType, 1),
        };
    }

    public Expr Visit(IShapeEvaluateContext context, Cast target) => context.GetArgumentShape(target, Cast.Input);

    private IRType Visit(Cast target, TensorType input)
    {
        return new TensorType(target.NewType, input.Shape);
    }
}
