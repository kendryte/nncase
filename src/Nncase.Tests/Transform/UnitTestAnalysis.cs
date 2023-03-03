// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.IO;
using System.Linq;
using Nncase.IR;
using Nncase.TIR;
using Nncase.Transform;
using Xunit;
using static Nncase.IR.F.Math;
using static Nncase.IR.F.Tensors;

namespace Nncase.Tests.TransformTest;

public sealed class UnitTestUsedByAnalysis : TestClassBase
{
    [Fact]
    public void TestMultiInput()
    {
        var input = new Var();
        var v0 = IR.F.Math.Unary(UnaryOp.Abs, input);
        var v1 = v0 + input;
        var v2 = IR.F.Math.Unary(UnaryOp.Abs, v1);

        var result = Analyser.AnalysisUsedBy(v2);
        Assert.Equal(2, result.Get(input).Count);
    }

    [Fact]
    public void TestMultInputWithFusion()
    {
        var fusionCase = new ReWrite.FusionTest.DataFlowType7FusionCaseLeft();
        var input = new Var("input", new TensorType(DataTypes.Float32, new int[] { 1, 3, 224, 224 }));
        var main = new Function(fusionCase.BuildBody(input), input);
        CompilerServices.InferenceType(main);

        var usedbyReslut = Analyser.AnalysisUsedBy(main);

        Assert.Equal(2, usedbyReslut.Get(input).Count);

        foreach (var k in usedbyReslut.MeMo.Keys)
        {
            if (k is Call { Target: Fusion { Name: "fusion_4_True" } })
            {
                Assert.Equal(2, usedbyReslut.Get(k).Count);
            }
        }
    }

    [Fact]
    public void TestMultInputWithTuple()
    {
        var input1 = new Var("input1", new TensorType(DataTypes.Float32, new int[] { 1, 3, 224, 224 }));
        var input2 = new Var("input2", new TensorType(DataTypes.Float32, new int[] { 1, 3, 224, 224 }));
        var v0 = IR.F.Tensors.Concat(new IR.Tuple(new[] { input1, input2 }), 1) + input2;
        var v1 = IR.F.Math.Quantize(v0, new QuantParam(1, 2.0f), DataTypes.UInt8);
        var main = new Function(v1, new[] { input1, input2 });
        CompilerServices.InferenceType(main);

        var usedbyReslut = Analyser.AnalysisUsedBy(main);

        Assert.Equal(1, usedbyReslut.Get(input1).Count);
        Assert.Equal(2, usedbyReslut.Get(input2).Count);
    }
}
