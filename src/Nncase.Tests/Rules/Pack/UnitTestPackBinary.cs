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
public class UnitTestPackBinary : TransformTestBase
{
    [Fact]
    public void TestPackBinaryPropagation()
    {
        var lhs = Testing.Rand<float>(1, 24);
        var lhsVar = new Var(new TensorType(lhs.ElementType, lhs.Shape));
        var rhs = Testing.Rand<float>(3, 1);
        var rhsVar = new Var(new TensorType(rhs.ElementType, rhs.Shape));
        Expr expr = lhsVar + rhsVar;
        expr = Pack(expr, [8], [1]);
        expr = Unpack(expr, [8], [1]);
        TestMatched<PackBinaryPropagation>(expr, new Dictionary<IVar, IValue> {
            { lhsVar, Value.FromTensor(lhs) },
            { rhsVar, Value.FromTensor(rhs) },
        });
    }
}
