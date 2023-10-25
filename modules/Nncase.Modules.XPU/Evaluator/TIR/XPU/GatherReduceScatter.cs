// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.TIR.XPU;

namespace Nncase.Evaluator.TIR.XPU;

public class GatherReduceScatterEvaluator : ITypeInferencer<GatherReduceScatter>
{
    public IRType Visit(ITypeInferenceContext context, GatherReduceScatter target)
    {
        _ = context.CheckArgumentType<IRType>(target, GatherReduceScatter.Input);
        _ = context.CheckArgumentType<IRType>(target, GatherReduceScatter.Output);
        return TupleType.Void;
    }
}
