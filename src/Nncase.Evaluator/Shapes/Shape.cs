// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Tensors;
using Nncase.Utilities;
using OrtKISharp;
using SMath = System.Math;

namespace Nncase.Evaluator;

internal partial class EvaluateVisitor
{
    protected override IValue VisitLeafShape(Shape expr)
    {
        var dims = expr.Dimensions.AsValueEnumerable().Select(GetDimValue).ToArray();
        return new ShapeValue(dims);
    }
}
