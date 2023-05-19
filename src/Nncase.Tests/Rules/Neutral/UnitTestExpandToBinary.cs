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
using Dimension = Nncase.IR.Dimension;
using Math = Nncase.IR.F.Math;
using Random = Nncase.IR.F.Random;
using Tensors = Nncase.IR.F.Tensors;

namespace Nncase.Tests.Rules.NeutralTest;

public class UnitTestExpandToBroadcast : TransformTestBase
{
    public static IEnumerable<object[]> TestExpandToBroadcastPositiveData =>
        new[]
        {
            new object[] { new[] { 3, 1 }, new[] { 2, 1, 6 } },
            new object[] { new[] { 3, 1 }, new[] { 3, 4 } },
            new object[] { new[] { 1, 256, 1, 1 }, new[] { 1, 256, 56, 56 } },
        };

    public static IEnumerable<object[]> TestExpandToBroadcastNegativeData =>
        new[]
        {
            new object[] { new[] { 2, 4, 8 }, new Var(new TensorType(DataTypes.Int32,  new IR.Shape(3))) },
        };

    [Theory]
    [MemberData(nameof(TestExpandToBroadcastPositiveData))]
    public void TestExpandToBroadcastPositive(int[] inputShape, int[] shape)
    {
        var a = Random.Normal(DataTypes.Float32, 0, 1, 0, inputShape);
        var rootPre = Tensors.Expand(a, shape);
        TestMatched<ExpandToBroadcast>(rootPre);
    }

    [Theory]
    [MemberData(nameof(TestExpandToBroadcastNegativeData))]
    public void TestExpandToBroadcastNegative(int[] inputShape, Expr shape)
    {
        var a = new IR.Var(new IR.TensorType(DataTypes.Float32, inputShape));
        var rootPre = Tensors.Squeeze(a, shape);
        TestNotMatch<ExpandToBroadcast>(rootPre);
    }
}
