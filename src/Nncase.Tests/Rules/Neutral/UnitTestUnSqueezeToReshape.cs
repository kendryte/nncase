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
using Tensorflow;
using Xunit;
using Math = Nncase.IR.F.Math;
using Random = Nncase.IR.F.Random;
using Tensors = Nncase.IR.F.Tensors;

namespace Nncase.Tests.Rules.NeutralTest;

public class UnitTestUnSqueezeToReshape : TransformTestBase
{
    public static IEnumerable<object[]> TestUnSqueezeToReshapePositiveData =>
        new[]
        {
            new object[] { new[] { 2, 4, 8 }, new[] { 0 } },
            new object[] { new[] { 4, 6 }, new[] { 0, 2 } },
            new object[] { new[] { 4, 6 }, new[] { -4, -2 } },
        };

    public static IEnumerable<object[]> TestUnSqueezeToReshapeNegativeData =>
        new[] { new object[] { new[] { 2, 4, IR.Dimension.Unknown }, new[] { -1 } }, };

    [Theory]
    [MemberData(nameof(TestUnSqueezeToReshapePositiveData))]
    public void TestUnSqueezeToReshapePositive(int[] shape, int[] axes)
    {
        var a = Random.Normal(DataTypes.Float32, 0, 1, 0, shape);
        var rootPre = Tensors.Unsqueeze(a, axes);
        TestMatched<UnSqueezeToReshape>(rootPre);
    }

    [Theory]
    [MemberData(nameof(TestUnSqueezeToReshapeNegativeData))]
    public void TestUnSqueezeToReshapeNegative(IR.Dimension[] shape, int[] axes)
    {
        var a = new IR.Var(new IR.TensorType(DataTypes.Float32, shape));
        var rootPre = Tensors.Unsqueeze(a, axes);
        TestNotMatch<UnSqueezeToReshape>(rootPre);
    }
}
