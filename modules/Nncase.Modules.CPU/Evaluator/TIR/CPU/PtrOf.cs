// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.TIR.CPU;

namespace Nncase.Evaluator.TIR.CPU;

public sealed class PtrOfEvaluator : ITypeInferencer<PtrOf>, IOpPrinter<PtrOf>
{
    public IRType Visit(ITypeInferenceContext context, PtrOf target) => new PointerType(target.DataType);

    public string Visit(IIRPrinterContext context, PtrOf target, bool iLmode)
    {
        if (iLmode)
        {
            throw new NotSupportedException();
        }

        return $"PtrOf({target.PtrName})";
    }
}
