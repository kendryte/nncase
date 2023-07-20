// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.Passes.Rules.Neutral;
using Nncase.Tests.TestFixture;
using Xunit;

namespace Nncase.Tests.Rules.NeutralTest;

[AutoSetupTestMethod(InitSession = true)]
public class UnitTestFoldNopIf : TransformTestBase
{
    [Fact]
    public void CondIsConst()
    {
        TestMatched<FoldNopIf>(new If(true, 1, 2));
        TestMatched<FoldNopIf>(new If(false, 1, 2));
    }

    [Fact]
    public void CondIsExpr()
    {
        var input = new Var(new TensorType(DataTypes.Boolean, Shape.Scalar));
        TestNotMatch<FoldNopIf>(new If(input, 1, 2));
    }
}
