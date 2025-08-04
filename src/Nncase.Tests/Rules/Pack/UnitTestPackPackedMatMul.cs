// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Generic;
using Nncase.IR;
using Nncase.Passes.Rules.Neutral;
using Nncase.Passes.Rules.NTT;
using Nncase.Tests.TestFixture;
using Xunit;
using static Nncase.IR.F.NTT;
using static Nncase.IR.F.Tensors;

namespace Nncase.Tests.Rules.NeutralTest;

[AutoSetupTestMethod(InitSession = true)]
public class UnitTestVectorizeVectorizedMatMul : TransformTestBase
{
    [Fact]
    public void TestVectorizedMatMulDevectorizePropagation()
    {
        var lhs = Vectorize(Testing.Rand<float>(3, 24), [8], [1]).Evaluate().AsTensor();
        var lhsVar = new Var(new TensorType(lhs.ElementType, lhs.Shape));
        var rhs = Vectorize(Testing.Rand<float>(24, 24), [8], [1]).Evaluate().AsTensor();
        var expr = Devectorize(lhsVar, [8], [1]);
        expr = VectorizedMatMul(expr, rhs, [], new int[] { 1 });
        expr = Devectorize(expr, [8], [1]);
        TestMatched<VectorizedMatMulDevectorizePropagation>(
            expr,
            new Dictionary<IVar, IValue> {
                { lhsVar, Value.FromTensor(lhs) },
            });
    }
}
