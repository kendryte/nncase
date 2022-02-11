// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using TorchSharp;
using Range = Nncase.IR.Tensors.Range;
using torchF = TorchSharp.torch.nn.functional;

namespace Nncase.Evaluator.Tensors;

/// <summary>
/// Evaluator for <see cref="Range"/>.
/// </summary>
public class RangeEvaluator : IEvaluator<Range>, ITypeInferencer<Range>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, Range range)
    {
        var begin = context.GetArgumentValueAsScalar<int>(range, Range.Begin);
        var end = context.GetArgumentValueAsScalar<int>(range, Range.End);
        var step = context.GetArgumentValueAsScalar<int>(range, Range.Step);
        return torch.arange(begin, end, step).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Range target)
    {
        if (context.GetArgument(target, Range.Begin) is TensorConst beginValue
            && context.GetArgument(target, Range.End) is TensorConst endValue
            && context.GetArgument(target, Range.Step) is TensorConst stepValue)
        {
            return new TensorType(
                DataType.Int32,
                new Shape((beginValue.Value.ToScalar<int>() + endValue.Value.ToScalar<int>()) / stepValue.Value.ToScalar<int>()));
        }

        return new InvalidType("Range begin, end, step should be constant");
    }
}
