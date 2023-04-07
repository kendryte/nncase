// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Nncase.Evaluator;
using Nncase.IR;
using Nncase.IR.F;
using Nncase.IR.Tensors;
using Nncase.Schedule;
using Nncase.Utilities;
using OrtKISharp;
using Xunit;
using static Nncase.IR.F.NN;
using static Nncase.IR.F.Tensors;
using static Nncase.Utilities.DumpUtility;
using Random = Nncase.IR.F.Random;

namespace Nncase.Tests.EvaluatorTest;

public class UnitTestEvaluatorBuffers : TestClassBase
{
    [Fact]
    public void TestUninitialized()
    {
        var shape = new[] { 1 };
        var expr = IR.F.Buffer.Uninitialized(DataTypes.Float32, MemoryLocation.Input, shape);
        CompilerServices.InferenceType(expr);
        Assert.Equal(Value.None, expr.Evaluate());
    }
}
