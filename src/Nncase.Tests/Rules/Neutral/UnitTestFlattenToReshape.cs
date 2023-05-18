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

public class UnitTestFlattenToReshape : TransformTestBase
{
    public static IEnumerable<object[]> TestFlattenToReshapePositiveData =>
        new[]
        {
            new object[] { new[] { 2, 2, 4, 8 }, 0 },
            new object[] { new[] { 3, 2, 4, 8 }, 1 },
            new object[] { new[] { 4, 4, 4, 6 }, 2 },
            new object[] { new[] { 5, 4, 4, 6 }, 3 },
            new object[] { new[] { 6, 2, 4, 8 }, -1 },
            new object[] { new[] { 7, 2, 4, 8 }, -2 },
            new object[] { new[] { 8, 4, 4, 6 }, -3 },
            new object[] { new[] { 9, 4, 4, 6 }, -4 },
        };

    public static IEnumerable<object[]> TestFlattenToReshapeNegativeData =>
        new[]
        {
            new object[] { new[] { 2, 4, IR.Dimension.Unknown }, 1 },
        };

    [Theory]
    [MemberData(nameof(TestFlattenToReshapePositiveData))]
    public void TestFlattenToReshapePositive(int[] shape, int axis)
    {
        var a = Random.Normal(DataTypes.Float32, 0, 1, 0, shape);
        var rootPre = Tensors.Flatten(a, axis);
        TestMatched<FlattenToReshape>(rootPre);
    }

    [Theory]
    [MemberData(nameof(TestFlattenToReshapeNegativeData))]
    public void TestFlattenToReshapeNegative(IR.Dimension[] shape, int axis)
    {
        var a = new IR.Var(new IR.TensorType(DataTypes.Float32, shape));
        var rootPre = Tensors.Flatten(a, axis);
        TestNotMatch<FlattenToReshape>(rootPre);
    }
}
