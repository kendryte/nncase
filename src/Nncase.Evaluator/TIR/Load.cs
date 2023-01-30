// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.IR;
using Nncase.TIR;

namespace Nncase.Evaluator.TIR;

/// <summary>
/// Evaluator for <see cref="Load"/>.
/// </summary>
public class LoadEvaluator : ITypeInferencer<Load>, IOpPrinter<Load>
{
    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Load target)
    {
        var handle = context.CheckArgumentType<TensorType>(target, Load.Handle);
        var index = context.CheckArgumentType<TensorType>(target, Load.Index);
        return Visit(target, handle, index);
    }

    /// <inheritdoc/>
    public string Visit(IIRPrinterContext context, Load target, bool iLmode)
    {
        var lhs = context.GetArgument(target, Load.Handle);
        var rhs = context.GetArgument(target, Load.Index);
        return $"{lhs}[{rhs}]";
    }

    private IRType Visit(Load target, TensorType handle, TensorType index)
    {
        if (!handle.IsScalar && handle.DType is not PointerType)
        {
            throw new NotSupportedException(handle.DType.ToString());
        }

        _ = index.IsScalar ? 1 : index.Shape[0].FixedValue;
        return TensorType.Scalar(((PointerType)handle.DType).ElemType);
    }
}
