// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Shapes;
using Nncase.IR.Tensors;
using Nncase.Utilities;
using OrtKISharp;
using SMath = System.Math;

namespace Nncase.Evaluator;

internal partial class EvaluateVisitor
{
    protected override IValue VisitLeafPadding(Padding expr)
    {
        var before = GetDimValue(expr.Before);
        var after = GetDimValue(expr.After);
        return new PaddingValue(before, after);
    }

    protected override IValue VisitLeafPaddings(Paddings expr)
    {
        var paddings = expr.Values.AsValueEnumerable().Select(x => (PaddingValue)Visit(x)).ToArray();
        return new PaddingsValue(paddings);
    }
}
