// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using OrtKISharp;

namespace Nncase.Evaluator.Math;

/// <summary>
/// Evaluator for <see cref="Clamp"/>.
/// </summary>
public class ClampEvaluator : IEvaluator<Clamp>, ITypeInferencer<Clamp>, ICostEvaluator<Clamp>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, Clamp clamp)
    {
        Console.WriteLine("Clamp Value");
        var input = context.GetOrtArgumentValue(clamp, Clamp.Input);
        Console.WriteLine(string.Join(",", input.ToArray<int>()));
        var min = context.GetOrtArgumentValue(clamp, Clamp.Min);
        Console.WriteLine(min.ToTensor().ToScalar<int>());
        var max = context.GetOrtArgumentValue(clamp, Clamp.Max);
        Console.WriteLine(max.ToTensor().ToScalar<int>());
        return OrtKI.Min(new[] { OrtKI.Max(new[] { input, min }), max }).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Clamp target)
    {
        var input = context.CheckArgumentType<TensorType>(target, Clamp.Input);
        var min = context.CheckArgumentType<TensorType>(target, Clamp.Min);
        var max = context.CheckArgumentType<TensorType>(target, Clamp.Max);
        if (input.DType != min.DType || input.DType != max.DType || min.DType != max.DType)
        {
            return new InvalidType(
                $"clamp type is not equal, input:{input.DType}, min:${min.DType}, max:${max.DType}");
        }

        return Visit(input, min, max);
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, Clamp target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, Clamp.Input);
        var minType = context.GetArgumentType<TensorType>(target, Clamp.Min);
        var maxType = context.GetArgumentType<TensorType>(target, Clamp.Max);
        var outputType = context.GetReturnType<TensorType>();

        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(inputType) + CostUtility.GetMemoryAccess(minType) + CostUtility.GetMemoryAccess(maxType),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(outputType),
            [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(outputType, 2),
        };
    }

    private IRType Visit(TensorType input, TensorType min, TensorType max)
    {
        if (TypeInference.BroadcastType(input, min) is InvalidType invalidMin)
        {
            return invalidMin;
        }

        if (TypeInference.BroadcastType(input, max) is InvalidType invalidMax)
        {
            return invalidMax;
        }

        if (min.Shape != max.Shape)
        {
            return new InvalidType($"The min.Shape {min.Shape} != max.Shape {max.Shape}");
        }

        return input;
    }
}
