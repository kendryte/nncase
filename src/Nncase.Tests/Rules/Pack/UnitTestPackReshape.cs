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
public class UnitTestPackReshape : TransformTestBase
{
    [Fact]
    public void TestPackReshapeFixedUnsqueeze()
    {
        var input = Testing.Rand<float>(3, 24);
        var inputVar = new Var(new TensorType(input.ElementType, input.Shape));
        var expr = Reshape(inputVar, [1, 3, 24]);
        TestMatchedCore(expr, new Dictionary<IVar, IValue> { { inputVar, Value.FromTensor(input) } }, new PackReshape(1, 32));
    }

    [Fact]
    public void TestPackReshapeDynamicUnsqueeze()
    {
        var dimX = new DimVar("x") { Metadata = { Range = (1, 256) } };
        var input = Testing.Rand<float>(3, 24);
        var inputVar = new Var(new TensorType(input.ElementType, [dimX, 24]));
        var expr = Reshape(inputVar, [1, dimX, 24]);
        TestMatchedCore(expr, new Dictionary<IVar, IValue> { { inputVar, Value.FromTensor(input) } }, new PackReshape(1, 32));
    }
}
