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

public sealed class UnitTestUsedByAnalysis : TestFixture.UnitTestFixtrue
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
        var passOptions = GetPassOptions(fusionCase.GetType().Name);
        var compileOptions = passOptions.CompileOptions;

        var target = CompilerServices.GetTarget(compileOptions.Target);
        var input = new Var("input", new TensorType(DataTypes.Float32, new int[] { 1, 3, 224, 224 }));
        var main = new Function(fusionCase.BuildBody(input), input);

        IRModule module = new(main);
        CompilerServices.InferenceType(main);
        CompilerServices.DumpIR(main, "pre", passOptions.DumpDir);

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
}