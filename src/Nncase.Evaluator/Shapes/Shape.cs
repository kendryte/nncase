// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using System.Reactive;
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
    protected override IValue VisitLeafRankedShape(RankedShape expr)
    {
        var dims = expr.Dimensions.AsValueEnumerable().Select(GetDimValue).ToArray();
        return new ShapeValue(dims);
    }

    protected override IValue VisitUnrankedShape(UnrankedShape expr)
    {
        var value = expr.Value;
        if (value is None)
        {
            throw new InvalidOperationException("UnrankedShape value is None");
        }

        return new ShapeValue(ExprMemo[value].AsTensor().Cast<long>().ToArray());
    }

    protected override IValue VisitShapeVar(ShapeVar expr)
    {
        if (!_varsValues.TryGetValue(expr, out var value))
        {
            throw new ArgumentException($"Must Set Input For Var {expr.Name}!");
        }

        if (value is ShapeValue)
        {
            return value;
        }

        throw new ArgumentException($"ShapeVar {expr.Name} must be a shape value!");
    }
}
