// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.CPU;

namespace Nncase.Evaluator.IR.CPU;

public sealed class LoadEvaluator : ITypeInferencer<Load>, ICostEvaluator<Load>
{
    public IRType Visit(ITypeInferenceContext context, Load target)
    {
        return context.GetArgumentType(target, Load.Input);
    }

    public Cost Visit(ICostEvaluateContext context, Load target) => new Cost()
    {
        [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(context.GetArgumentType<IRType>(target, Load.Input)),
        [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(context.GetArgumentType<IRType>(target, Load.Input)),
    };
}
