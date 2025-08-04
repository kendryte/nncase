// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Generic;
using Nncase.IR;
using Nncase.Passes.Rules.Neutral;
using Nncase.Passes.Rules.NTT;
using Nncase.Tests.TestFixture;
using Xunit;
using static Nncase.IR.F.Tensors;

namespace Nncase.Tests.Rules.NeutralTest;

[AutoSetupTestMethod(InitSession = true)]
public class UnitTestVectorizeReshape : TransformTestBase
{
    [Fact]
    public void TestVectorizeReshapeFixedUnsqueeze()
    {
        var input = Testing.Rand<float>(3, 24);
        var inputVar = new Var(new TensorType(input.ElementType, input.Shape));
        var expr = Reshape(inputVar, [1, 3, 24]);
        TestMatchedCore(expr, new Dictionary<IVar, IValue> { { inputVar, Value.FromTensor(input) } }, new VectorizeReshape(1, 32));
    }

    [Fact]
    public void TestVectorizeReshapeDynamicUnsqueeze()
    {
        var dimX = new DimVar("x") { Metadata = { Range = (1, 256) } };
        var input = Testing.Rand<float>(3, 24);
        var inputVar = new Var(new TensorType(input.ElementType, [dimX, 24]));
        var expr = Reshape(inputVar, [1, dimX, 24]);
        TestMatchedCore(expr, new Dictionary<IVar, IValue> { { inputVar, Value.FromTensor(input) } }, new VectorizeReshape(1, 32));
    }

    [Fact]
    public void TestVectorizeReshapePropagationDynamicUnsqueeze()
    {
        var dimX = new DimVar("x") { Metadata = { Range = (1, 256) } };
        var input = Testing.Rand<float>(3, 24);
        var inputVar = new Var(new TensorType(input.ElementType, [dimX, 24]));
        Expr expr = Reshape(inputVar, [1, dimX, 24]);
        expr = Vectorize(expr, [8], [2]);
        expr = Devectorize(expr, [8], [2]);
        TestMatched<VectorizeReshapePropagation>(expr, new Dictionary<IVar, IValue> { { inputVar, Value.FromTensor(input) } });
    }

    [Fact]
    public void TestReshapeDevectorizePropagationDynamicUnsqueeze()
    {
        var dimX = new DimVar("x") { Metadata = { Range = (1, 256) } };
        var input = Vectorize(Testing.Rand<float>(3, 24), [8], [1]).Evaluate().AsTensor();
        var inputVar = new Var(new TensorType(input.ElementType, [dimX, 3]));
        Expr expr = Reshape(Devectorize(inputVar, [8], [1]), [1, dimX, 24]);
        TestMatched<ReshapeDevectorizePropagation>(expr, new Dictionary<IVar, IValue> { { inputVar, Value.FromTensor(input) } });
    }

    [Fact]
    public void TestReshapeDevectorizePropagationDynamicCombine()
    {
        var dimX = new DimVar("x") { Metadata = { Range = (1, 256) } };
        var input = Vectorize(Testing.Rand<float>(3, 3, 24), [8], [2]).Evaluate().AsTensor();
        var inputVar = new Var(new TensorType(input.ElementType, [dimX, 3, 3]));
        Expr expr = Reshape(Devectorize(inputVar, [8], [2]), [1, dimX, 72]);
        TestMatched<ReshapeDevectorizePropagation>(expr, new Dictionary<IVar, IValue> { { inputVar, Value.FromTensor(input) } });
    }

    [Fact]
    public void TestReshapeDevectorizePropagationDynamicSplit()
    {
        var dimX = new DimVar("x") { Metadata = { Range = (1, 256) } };
        var input = Vectorize(Testing.Rand<float>(3, 72), [8], [1]).Evaluate().AsTensor();
        var inputVar = new Var(new TensorType(input.ElementType, [dimX, 9]));
        Expr expr = Reshape(Devectorize(inputVar, [8], [1]), [1, dimX, 3, 24]);
        TestMatched<ReshapeDevectorizePropagation>(expr, new Dictionary<IVar, IValue> { { inputVar, Value.FromTensor(input) } });
    }
}
