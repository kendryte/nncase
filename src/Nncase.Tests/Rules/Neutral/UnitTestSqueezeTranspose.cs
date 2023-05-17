// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Nncase.IR;
using Nncase.IR.F;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using Nncase.Passes;
using Nncase.Passes.Rules.Neutral;
using Nncase.PatternMatch;
using Nncase.Tests.TestFixture;
using Xunit;
using static Nncase.IR.F.NN;
using Math = Nncase.IR.F.Math;
using Random = Nncase.IR.F.Random;
using Tensors = Nncase.IR.F.Tensors;
using Tuple = System.Tuple;

namespace Nncase.Tests.Rules.NeutralTest;

[AutoSetupTestMethod(InitSession = true)]
public class UnitTestSqueezeTranspose : TransformTestBase
{
    public static readonly TheoryData<int[]> Squeeze5DTransposePositiveData =
        new()
        {
            new[] { 0, 1, 2, 3, 4 },
            new[] { 0, 1, 2, 4, 3 },
            new[] { 0, 1, 3, 2, 4 },
            new[] { 0, 1, 3, 4, 2 },
            new[] { 0, 1, 4, 2, 3 },
            new[] { 0, 1, 4, 3, 2 },
            new[] { 0, 2, 1, 3, 4 },
            new[] { 0, 2, 1, 4, 3 },
            new[] { 0, 2, 3, 1, 4 },
            new[] { 0, 2, 3, 4, 1 },
            new[] { 0, 2, 4, 1, 3 },
            new[] { 0, 2, 4, 3, 1 },
            new[] { 0, 3, 1, 2, 4 },
            new[] { 0, 3, 1, 4, 2 },
            new[] { 0, 3, 2, 1, 4 },
            new[] { 0, 3, 2, 4, 1 },
            new[] { 0, 3, 4, 1, 2 },
            new[] { 0, 3, 4, 2, 1 },
            new[] { 0, 4, 1, 2, 3 },
            new[] { 0, 4, 1, 3, 2 },
            new[] { 0, 4, 2, 1, 3 },
            new[] { 0, 4, 2, 3, 1 },
            new[] { 0, 4, 3, 1, 2 },
            new[] { 0, 4, 3, 2, 1 },
            new[] { 1, 0, 2, 3, 4 },
            new[] { 1, 0, 2, 4, 3 },
            new[] { 1, 0, 3, 2, 4 },
            new[] { 1, 0, 3, 4, 2 },
            new[] { 1, 0, 4, 2, 3 },
            new[] { 1, 0, 4, 3, 2 },
            new[] { 1, 2, 0, 3, 4 },
            new[] { 1, 2, 0, 4, 3 },
            new[] { 1, 2, 3, 0, 4 },
            new[] { 1, 2, 3, 4, 0 },
            new[] { 1, 2, 4, 0, 3 },
            new[] { 1, 2, 4, 3, 0 },
            new[] { 1, 3, 0, 2, 4 },
            new[] { 1, 3, 0, 4, 2 },
            new[] { 1, 3, 2, 0, 4 },
            new[] { 1, 3, 2, 4, 0 },
            new[] { 1, 3, 4, 0, 2 },
            new[] { 1, 3, 4, 2, 0 },
            new[] { 1, 4, 0, 2, 3 },
            new[] { 1, 4, 0, 3, 2 },
            new[] { 1, 4, 2, 0, 3 },
            new[] { 1, 4, 2, 3, 0 },
            new[] { 1, 4, 3, 0, 2 },
            new[] { 1, 4, 3, 2, 0 },
            new[] { 2, 0, 1, 3, 4 },
            new[] { 2, 0, 1, 4, 3 },
            new[] { 2, 0, 3, 1, 4 },
            new[] { 2, 0, 3, 4, 1 },
            new[] { 2, 0, 4, 1, 3 },
            new[] { 2, 0, 4, 3, 1 },
            new[] { 2, 1, 0, 3, 4 },
            new[] { 2, 1, 0, 4, 3 },
            new[] { 2, 1, 3, 0, 4 },
            new[] { 2, 1, 3, 4, 0 },
            new[] { 2, 1, 4, 0, 3 },
            new[] { 2, 1, 4, 3, 0 },
            new[] { 2, 3, 0, 1, 4 },
            new[] { 2, 3, 0, 4, 1 },
            new[] { 2, 3, 1, 0, 4 },
            new[] { 2, 3, 1, 4, 0 },
            new[] { 2, 3, 4, 0, 1 },
            new[] { 2, 3, 4, 1, 0 },
            new[] { 2, 4, 0, 1, 3 },
            new[] { 2, 4, 0, 3, 1 },
            new[] { 2, 4, 1, 0, 3 },
            new[] { 2, 4, 1, 3, 0 },
            new[] { 2, 4, 3, 0, 1 },
            new[] { 2, 4, 3, 1, 0 },
            new[] { 3, 0, 1, 2, 4 },
            new[] { 3, 0, 1, 4, 2 },
            new[] { 3, 0, 2, 1, 4 },
            new[] { 3, 0, 2, 4, 1 },
            new[] { 3, 0, 4, 1, 2 },
            new[] { 3, 0, 4, 2, 1 },
            new[] { 3, 1, 0, 2, 4 },
            new[] { 3, 1, 0, 4, 2 },
            new[] { 3, 1, 2, 0, 4 },
            new[] { 3, 1, 2, 4, 0 },
            new[] { 3, 1, 4, 0, 2 },
            new[] { 3, 1, 4, 2, 0 },
            new[] { 3, 2, 0, 1, 4 },
            new[] { 3, 2, 0, 4, 1 },
            new[] { 3, 2, 1, 0, 4 },
            new[] { 3, 2, 1, 4, 0 },
            new[] { 3, 2, 4, 0, 1 },
            new[] { 3, 2, 4, 1, 0 },
            new[] { 3, 4, 0, 1, 2 },
            new[] { 3, 4, 0, 2, 1 },
            new[] { 3, 4, 1, 0, 2 },
            new[] { 3, 4, 1, 2, 0 },
            new[] { 3, 4, 2, 0, 1 },
            new[] { 3, 4, 2, 1, 0 },
            new[] { 4, 0, 1, 2, 3 },
            new[] { 4, 0, 1, 3, 2 },
            new[] { 4, 0, 2, 1, 3 },
            new[] { 4, 0, 2, 3, 1 },
            new[] { 4, 0, 3, 1, 2 },
            new[] { 4, 0, 3, 2, 1 },
            new[] { 4, 1, 0, 2, 3 },
            new[] { 4, 1, 0, 3, 2 },
            new[] { 4, 1, 2, 0, 3 },
            new[] { 4, 1, 2, 3, 0 },
            new[] { 4, 1, 3, 0, 2 },
            new[] { 4, 1, 3, 2, 0 },
            new[] { 4, 2, 0, 1, 3 },
            new[] { 4, 2, 0, 3, 1 },
            new[] { 4, 2, 1, 0, 3 },
            new[] { 4, 2, 1, 3, 0 },
            new[] { 4, 2, 3, 0, 1 },
            new[] { 4, 2, 3, 1, 0 },
            new[] { 4, 3, 0, 1, 2 },
            new[] { 4, 3, 0, 2, 1 },
            new[] { 4, 3, 1, 0, 2 },
            new[] { 4, 3, 1, 2, 0 },
            new[] { 4, 3, 2, 0, 1 },
            new[] { 4, 3, 2, 1, 0 },
        };

    [Theory]
    [MemberData(nameof(Squeeze5DTransposePositiveData))]
    public void TestSqueeze5DTransposePositive(int[] perm)
    {
        var input = Random.Normal(DataTypes.Float32, 0, 1, 1, new[] { 1, 3, 24, 56, 32 });

        var rootPre = Tensors.Transpose(input, perm);
        TestMatched<Squeeze5DTranspose>(rootPre);
    }
}
