using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.IO;
using Nncase.IR;
using Nncase.Transform;
using Xunit;
using static Nncase.IR.F.Tensors;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Tests.ReWrite.FusionTest;

public class UnitTestEGraphFusion : TestFixture.UnitTestFixtrue
{
    [Fact]
    public void TestResNet18Fusion()
    {
        var passOptions = GetPassOptions();
        var compileOptions = passOptions.CompileOptions;

        var target = CompilerServices.GetTarget(compileOptions.Target);

        // step 1. import
        var input = new Var("input", new TensorType(DataTypes.Float32, new int[] { 1, 3, 224, 224 }));
        var model = new ResNet(typeof(BasicBlock), new[] { 2, 2, 2, 2 });
        var body = model.Forward(input);
        var main = new Function("main", body, ImmutableArray.Create(input));
        IRModule module = new(main);

        CompilerServices.InferenceType(main);
        CompilerServices.DumpIR(main, "", passOptions.DumpDir);
    }

}