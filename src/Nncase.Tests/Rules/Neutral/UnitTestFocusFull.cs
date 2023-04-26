// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.Passes;
using Nncase.Passes.Rules.Neutral;
using Xunit;

namespace Nncase.Tests.Rules.NeutralTest;

[TestFixture.AutoSetupTestMethod(InitSession = true)]
public class UnitTestFocusFull : TransformTestBase
{
    [Fact]
    public void TestPositive()
    {
        var shape = new[] { 1, 3, 640, 640 };
        var input = new Var("input", new TensorType(DataTypes.Float32, shape));
        var feedDict = new Dictionary<Var, IValue> { { input, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 0, shape).Evaluate() } };

        var s0 = IR.F.Tensors.Slice(input, new[] { 0, 0 }, new[] { 640, 640 }, new[] { 2, 3 }, new[] { 2, 2 });
        var s1 = IR.F.Tensors.Slice(input, new[] { 1, 0 }, new[] { 640, 640 }, new[] { 2, 3 }, new[] { 2, 2 });
        var s2 = IR.F.Tensors.Slice(input, new[] { 0, 1 }, new[] { 640, 640 }, new[] { 2, 3 }, new[] { 2, 2 });
        var s3 = IR.F.Tensors.Slice(input, new[] { 1, 1 }, new[] { 640, 640 }, new[] { 2, 3 }, new[] { 2, 2 });
        var rootPre = IR.F.Tensors.Concat(new IR.Tuple(s0, s1, s2, s3), 1);
        TestMatched<FocusFull>(rootPre, feedDict);
    }

    [Fact]
    public void TestNegative()
    {
        var shape = new[] { 1, 3, 640, 640 };
        var input = new Var("input", new TensorType(DataTypes.Float32, shape));
        var s0 = IR.F.Tensors.Slice(input, new[] { 0, 0 }, new[] { 640, 640 }, new[] { 2, 3 }, new[] { 2, 2 });
        var s1 = IR.F.Tensors.Slice(input, new[] { 1, 0 }, new[] { 640, 640 }, new[] { 2, 3 }, new[] { 2, 2 });
        var s2 = IR.F.Tensors.Slice(input, new[] { 0, 1 }, new[] { 640, 280 }, new[] { 2, 3 }, new[] { 2, 2 });
        var s3 = IR.F.Tensors.Slice(input, new[] { 1, 1 }, new[] { 640, 640 }, new[] { 2, 3 }, new[] { 2, 2 });
        var rootPre = IR.F.Tensors.Concat(new IR.Tuple(s0, s1, s2, s3), 1);
        TestNotMatch<FocusFull>(rootPre);
    }
}
