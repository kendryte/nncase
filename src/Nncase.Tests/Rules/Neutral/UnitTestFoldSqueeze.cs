// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Generic;
using Nncase.IR;
using Nncase.Passes.Rules.Neutral;
using Nncase.Tests.TestFixture;
using Xunit;
using static Nncase.IR.F.Tensors;

namespace Nncase.Tests.Rules.NeutralTest;

[AutoSetupTestMethod(InitSession = true)]
public class UnitTestFoldSqueeze : TransformTestBase
{
    [Fact]
    public void TestFoldSqueezeUnsqueeze()
    {
        var input = Testing.Rand<float>(1, 3, 24);
        var inputVar = new Var(new TensorType(input.ElementType, input.Shape));
        var expr = Squeeze(Unsqueeze(inputVar, new[] { -3 }), new[] { 1 });
        TestMatched<FoldSqueezeUnsqueeze>(expr, new Dictionary<Var, IValue> { { inputVar, Value.FromTensor(input) } });
    }

    [Fact]
    public void TestFoldUnsqueezeSqueeze()
    {
        var input = Testing.Rand<float>(1, 1, 3, 24);
        var inputVar = new Var(new TensorType(input.ElementType, input.Shape));
        var expr = Unsqueeze(Squeeze(inputVar, new[] { 1 }), new[] { -3 });
        TestMatched<FoldUnsqueezeSqueeze>(expr, new Dictionary<Var, IValue> { { inputVar, Value.FromTensor(input) } });
    }
}
