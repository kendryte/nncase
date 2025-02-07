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
using Nncase.Tests.TestFixture;
using Xunit;
using static Nncase.IR.F.Tensors;
using Math = Nncase.IR.F.Math;
using NN = Nncase.IR.F.NN;
using Random = Nncase.IR.F.Random;

namespace Nncase.Tests.Rules.NeutralTest;

[AutoSetupTestMethod(InitSession = true)]
public class UnitTestSpaceToBatchToPad : TransformTestBase
{
    public static IEnumerable<object[]> TestSpaceToBatchToPadPositiveData =>
        new[]
        {
            new object[] { new[] { 1, 128, 128, 3 }, new[] { 1, 1 }, new[,] { { 1, 1 }, { 1, 1 } } },
            new object[] { new[] { 3, 64, 64, 16 }, new[] { 1, 1 }, new[,] { { 2, 2 }, { 0, 3 } } },
            new object[] { new[] { 3, 32, 32, 16 }, new[] { 1, 1 }, new[,] { { 3, 8 }, { 7, 4 } } },
        };

    public static IEnumerable<object[]> TestSpaceToBatchToPadNegativeData =>
        new[]
        {
            new object[] { new[] { 1, 128, 128, new IR.Dimension(1) }, new[] { 2, 2 }, new[,] { { 0, 0 }, { 0, 0 } } },
            new object[] { new[] { 1, 128, 128, IR.Dimension.Unknown }, new[] { 1, 1 }, new[,] { { 1, 1 }, { 1, 1 } } },
        };

    [Theory]
    [MemberData(nameof(TestSpaceToBatchToPadPositiveData))]
    public void TestSpaceToBatchToPadPositive(int[] shape, int[] blockShape, int[,] paddings)
    {
        var a = Random.Normal(DataTypes.Float32, 0, 1, 0, shape);
        var rootPre = NCHWToNHWC(NN.SpaceToBatch(NHWCToNCHW(a), blockShape, paddings));
        TestMatched<SpaceToBatchToPad>(rootPre);
    }

    [Theory]
    [MemberData(nameof(TestSpaceToBatchToPadNegativeData))]
    public void TestFlattenToReshapeNegative(IR.Dimension[] shape, int[] blockShape, int[,] paddings)
    {
        var a = new IR.Var(new IR.TensorType(DataTypes.Float32, shape));
        var rootPre = NN.SpaceToBatch(a, blockShape, paddings);
        TestNotMatch<SpaceToBatchToPad>(rootPre);
    }
}
