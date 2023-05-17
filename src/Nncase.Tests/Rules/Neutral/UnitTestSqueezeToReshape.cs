// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using Nncase.Passes;
using Nncase.Passes.Rules.Neutral;
using Xunit;
using Math = Nncase.IR.F.Math;
using Random = Nncase.IR.F.Random;
using Tensors = Nncase.IR.F.Tensors;

namespace Nncase.Tests.Rules.NeutralTest;

public class UnitTestSqueezeToReshape : TransformTestBase
{
    public static IEnumerable<object[]> TestSqueezeToReshapePositiveData =>
        new[]
        {
            new object[] { new[] { 1, 2, 4, 8 }, new[] { 0 } },
            new object[] { new[] { 2, 1, 1, 8 }, Array.Empty<int>() },
            new object[] { new[] { 1, 4, 1, 6 }, new[] { 0, 2 } },
            new object[] { new[] { 1, 4, 1, 6 }, new[] { -4, -2 } },
        };

    public static IEnumerable<object[]> TestSqueezeToReshapeNegativeData =>
        new[]
        {
            new object[] { new[] { 2, 4, IR.Dimension.Unknown }, Array.Empty<int>() },
        };

    [Theory]
    [MemberData(nameof(TestSqueezeToReshapePositiveData))]
    public void TestSqueezeToReshapePositive(int[] shape, int[] axes)
    {
        var a = Random.Normal(DataTypes.Float32, 0, 1, 0, shape);
        var rootPre = Tensors.Squeeze(a, axes);
        TestMatched<SqueezeToReshape>(rootPre);
    }

    [Theory]
    [MemberData(nameof(TestSqueezeToReshapeNegativeData))]
    public void TestSqueezeToReshapeNegative(IR.Dimension[] shape, int[] axes)
    {
        var a = new IR.Var(new IR.TensorType(DataTypes.Float32, shape));
        var rootPre = Tensors.Squeeze(a, axes);
        TestNotMatch<SqueezeToReshape>(rootPre);
    }
}
