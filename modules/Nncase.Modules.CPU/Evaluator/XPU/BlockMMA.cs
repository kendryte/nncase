// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.XPU;

namespace Nncase.Evaluator.XPU;

public sealed class BlockMMAEvaluator : ITypeInferencer<BlockMMA>
{
    public IRType Visit(ITypeInferenceContext context, BlockMMA target) => TupleType.Void;
}
