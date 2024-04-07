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

public sealed class StoreEvaluator : ITypeInferencer<Store>, ICostEvaluator<Store>
{
    public IRType Visit(ITypeInferenceContext context, Store target)
    {
        return context.GetArgumentType(target, Store.Input);
    }

    public Cost Visit(ICostEvaluateContext context, Store target) => new Cost()
    {
        [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(context.GetArgumentType<IRType>(target, Store.Input)),
        [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(context.GetArgumentType<IRType>(target, Store.Input)),
    };
}
