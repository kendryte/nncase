// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.IO;
using System.Linq;
using System.Reactive;
using NetFabric.Hyperlinq;
using Nncase.CostModel;
using Nncase.Evaluator;
using Nncase.IR;
using Nncase.IR.F;
using Nncase.IR.Math;
using Nncase.IR.Random;
using Nncase.IR.Tensors;
using Nncase.TIR;
using Nncase.Utilities;
using OrtKISharp;
using Xunit;
using static Nncase.IR.F.NN;
using static Nncase.IR.F.Tensors;
using static Nncase.Utilities.DumpUtility;
using Tuple = Nncase.IR.Tuple;

namespace Nncase.Tests.EvaluatorTest;

public class UnitTestCostEvaluate : ExprVisitor<Cost, Unit>
{
    [Fact]
    public void TestCostEvaluateContext()
    {
        var costEvaluateContext = new CostEvaluateContext(ExprMemo);
        Assert.Throws<InvalidOperationException>(() => costEvaluateContext.CurrentCall);
        var loadCall = new Call(new Load());
        costEvaluateContext.CurrentCall = loadCall;
        Assert.Equal(loadCall, costEvaluateContext.CurrentCall);
    }

    [Fact]
    public void TestCostEvaluateVisitor()
    {
        var costEvaluateVisitor1 = new CostEvaluateVisitor();
        var costEvaluateVisitor2 = new CostEvaluateVisitor();
        Assert.False(costEvaluateVisitor1.Equals(costEvaluateVisitor2));
    }
}
