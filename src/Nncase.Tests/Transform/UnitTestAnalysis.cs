// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.IO;
using System.Linq;
using DryIoc.ImTools;
using Microsoft.Extensions.DependencyInjection;
using Nncase.IR;
using Nncase.Passes;
using Nncase.Passes.Analysis;
using Nncase.Tests.TestFixture;
using Nncase.TIR;
using Xunit;
using static Nncase.IR.F.Math;
using static Nncase.IR.F.Tensors;

namespace Nncase.Tests.TransformTest;

[AutoSetupTestMethod(InitSession = true)]
public sealed class UnitTestAnalysis : TestClassBase
{
    public IAnalyzerManager AnalyzerMananger => CompileSession.GetRequiredService<IAnalyzerManager>();

    [Fact]
    public void TestMultiInput()
    {
        var input = new Var();
        var v0 = IR.F.Math.Unary(UnaryOp.Abs, input);
        var v1 = v0 + input;
        var v2 = IR.F.Math.Unary(UnaryOp.Abs, v1);
        var func = new Function(v2, input);

        var userAnalysis = AnalyzerMananger.GetAnaylsis<IExprUserAnalysisResult>(func);
        Assert.Equal(2, userAnalysis[input].Count());
    }

    [Fact]
    public void TestMultInputWithFusion()
    {
        var fusionCase = new ReWrite.FusionTest.DataFlowType7FusionCaseLeft();
        var input = new Var("input", new TensorType(DataTypes.Float32, new int[] { 1, 3, 224, 224 }));
        var main = new Function(fusionCase.BuildBody(input), input);
        CompilerServices.InferenceType(main);

        var userAnalysis = AnalyzerMananger.GetAnaylsis<IExprUserAnalysisResult>(main);

        Assert.Equal(2, userAnalysis[input].Count());
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

        var userAnalysis = AnalyzerMananger.GetAnaylsis<IExprUserAnalysisResult>(main);

        Assert.Single(userAnalysis[input1]);
        Assert.Equal(2, userAnalysis[input2].Count());
    }
}
