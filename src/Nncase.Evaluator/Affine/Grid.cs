// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.Affine;

namespace Nncase.Evaluator;

internal sealed partial class TypeInferenceVisitor
{
    protected override IRType VisitLeafGrid(Grid expr) => expr.Buffers[^1].CheckedType;
}
