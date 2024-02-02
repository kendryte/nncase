// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Ncnn;
using OrtKISharp;

namespace Nncase.Evaluator.Ncnn;

/// <summary>
/// Evaluator for <see cref="NcnnPadding"/>.
/// </summary>
public class NcnnPaddingEvaluator : IEvaluator<NcnnPadding>, ITypeInferencer<NcnnPadding>, ICostEvaluator<NcnnPadding>, IShapeEvaluator<NcnnPadding>, IMetricEvaluator<NcnnPadding>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, NcnnPadding padding)
    {
        var input = context.GetOrtArgumentValue(padding, NcnnPadding.Input);
        // ncnn not support N
        var pads = new Tensor<long>(new int[] { 0, 0, padding.Front, padding.Top, padding.Left, padding.Behind, padding.Bottom, padding.Right });
        return OrtKI.Pad(input, pads.ToOrtTensor(), padding.Value, padding.Type switch
        {
            0 => "Constant",
            1 => "Reflect",
            2 => "Edge",
            _ => "Symmetric",
        }).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, NcnnPadding target)
    {
        var input = context.CheckArgumentType<TensorType>(target, NcnnPadding.Input);
        var c = target.Front + target.Behind;
        var h = target.Left + target.Right;
        var w = target.Top + target.Bottom;
        return Visit(input, c, h, w);
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, NcnnPadding target)
    {
        var ret = context.GetReturnType<TensorType>();
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(ret),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(ret),
        };
    }

    public Metric Visit(IMetricEvaluateContext context, NcnnPadding target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, NcnnPadding.Input);
        var outputType = context.GetReturnType<IRType>();

        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(inputType),
            [CostFactorNames.MemoryStore] = outputType is TensorType outT ? CostUtility.GetMemoryAccess(outT) : CostUtility.GetMemoryAccess(inputType),
        };
    }

    public Expr Visit(IShapeEvaluateContext context, NcnnPadding target) => context.GetArgumentShape(target, NcnnPadding.Input);

    private IRType Visit(TensorType input, int c, int h, int w)
    {
        if (input.Shape.Count != 3)
        {
            throw new ArgumentException($"{nameof(NcnnPadding)} only supports 3-dims input.");
        }

        var newShape = new int[] { 0, 0, 0 }; // Must be 3-dims without BatchSize.
        newShape[0] = input.Shape[0].FixedValue + c;
        newShape[1] = input.Shape[1].FixedValue + h;
        newShape[2] = input.Shape[2].FixedValue + w;
        return new TensorType(input.DType, newShape);
    }
}
