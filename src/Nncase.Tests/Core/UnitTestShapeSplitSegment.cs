// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.Extensions.DependencyInjection;
using Nncase.CodeGen;
using Nncase.IR;
using Nncase.Tests.TestFixture;
using Xunit;

namespace Nncase.Tests.CoreTest;

[AutoSetupTestMethod(InitSession = true)]
public class UnitTestShapeSplitSegment : TestClassBase
{
    [Theory]
    [InlineData(64, 112)]
    [InlineData(112, 112)]
    [InlineData(190, 224)]
    [InlineData(224, 224)]
    public void TestSimpleShapeSplit(int dim, int expectDim)
    {
        var inVar = new Var(new TensorType(DataTypes.Float32, new[] { 1, 3, Dimension.Unknown, 224 }));
        var f = new Function((inVar * 2f) + 1f, inVar);
        var info = new SegmentInfo(0, 2, new[] { 112, 224 });
        var module = new ShapeSplitSegment().Run(f, info);
        var (_, kmodel) = Testing.BuildKModel("TestSimple.kmodel", module, CompileSession);
        var input = new[] { Testing.Rand<float>(1, 3, dim, 224) };
        var actual = Testing.RunKModel(kmodel, Dumpper.Directory, input).AsTensors()[0];
        Assert.Equal(expectDim, actual.Shape[2]);
        var expect = f.Body.Evaluate(new Dictionary<Var, IValue> { { inVar, Value.FromTensor(input[0]) } }).AsTensors()[0];
        if (dim == expectDim)
        {
            Assert.True(Comparator.CosSimilarity(expect, actual) > 0.99);
        }
    }

    [Fact]
    public void TestOutOfSegment()
    {
        var inVar = new Var(new TensorType(DataTypes.Float32, new[] { 1, 3, Dimension.Unknown, 224 }));
        var f = new Function(inVar, new[] { inVar });
        var info = new SegmentInfo(0, 2, new[] { 112, 224 });
        var module = new ShapeSplitSegment().Run(f, info);
        var (_, kmodel) = Testing.BuildKModel("TestSimple.kmodel", module, CompileSession);
        var input = new[] { Testing.Rand<float>(1, 3, 225, 224) };
        Assert.Throws<InvalidOperationException>(
            () => Testing.RunKModel(kmodel, Dumpper.Directory, input));
    }
}
