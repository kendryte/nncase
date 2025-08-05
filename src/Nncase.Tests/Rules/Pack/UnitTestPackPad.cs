// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Generic;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.Passes;
using Nncase.Passes.Rules.Neutral;
using Nncase.Passes.Rules.NTT;
using Nncase.Tests.TestFixture;
using Xunit;
using static Nncase.IR.F.NN;
using static Nncase.IR.F.Tensors;

namespace Nncase.Tests.Rules.NeutralTest;

[AutoSetupTestMethod(InitSession = true)]
public class UnitTestVectorizePad : TransformTestBase
{
    [Fact]
    public void TestVectorizePadPropagationFixed()
    {
        var input = Testing.Rand<float>(3, 24);
        var inputVar = new Var(new TensorType(input.ElementType, input.Shape));
        Expr expr = Pad(inputVar, new([(0, 0), (8, 16)]), PadMode.Constant, 0f);
        expr = Pack(expr, [8], [1]);
        expr = Unpack(expr, [8], [1]);
        TestMatched<VectorizePadPropagation>(expr, new Dictionary<IVar, IValue> { { inputVar, Value.FromTensor(input) } });
    }

    [Fact]
    public void TestVectorizePadPropagationDynamic()
    {
        var dimX = new DimVar("x") { Metadata = { Range = (1, 256) } };
        var input = Testing.Rand<float>(3, 24);
        var inputVar = new Var(new TensorType(input.ElementType, [dimX, 24]));
        Expr expr = Pad(inputVar, new([(0, 0), (8, 16)]), PadMode.Constant, 0f);
        expr = Pack(expr, [8], [1]);
        expr = Unpack(expr, [8], [1]);
        TestMatched<VectorizePadPropagation>(expr, new Dictionary<IVar, IValue> { { inputVar, Value.FromTensor(input) } });
    }

    [Fact]
    public void TestVectorizePadPropagationQwen3()
    {
        CompileOptions.DumpFlags = DumpFlags.PassIR | DumpFlags.Rewrite;
        var sequenceLength = new DimVar("sequence_length") { Metadata = { Range = (1, 256) } };
        var input = Testing.Rand<float>(16, 3, 128);
        var inputVar = new Var(new TensorType(input.ElementType, [16, sequenceLength, 128]));
        Expr expr = Pad(inputVar, new([(0, 0), (0, 1024 - sequenceLength), (0, 0)]), PadMode.Constant, 0f);
        expr = Pack(expr, [32], [2]);
        expr = Unpack(expr, [32], [2]);
        var func = new Function("main", expr, [inputVar]);
        var module = new IRModule(func);

        var pmgr = CompileSession.CreatePassManager("Vectorize");
        pmgr.Add<EGraphRulesPass>()
            .Configure(c =>
            {
                c.Add<VectorizePadPropagation>();
            });
        pmgr.RunAsync(module).Wait();
        Assert.True(module.Entry is Function { Body: Call { Target: IR.Tensors.Unpack, Arguments: var devectorizeArgs } }
            && devectorizeArgs[0] is Call { Target: IR.NN.Pad });
    }
}
