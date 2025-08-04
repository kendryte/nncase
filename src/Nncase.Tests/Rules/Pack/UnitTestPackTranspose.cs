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
public class UnitTestVectorizeTranspose : TransformTestBase
{
    [Fact]
    public void TestVectorizeTransposePropagation()
    {
        var input = Testing.Rand<float>(3, 24);
        var inputVar = new Var(new TensorType(input.ElementType, input.Shape));
        Expr expr = Transpose(inputVar, [1, 0]);
        expr = Vectorize(expr, [8], [0]);
        expr = Devectorize(expr, [8], [0]);
        TestMatched<VectorizeTransposePropagation>(expr, new Dictionary<IVar, IValue> { { inputVar, Value.FromTensor(input) } });
    }
}
