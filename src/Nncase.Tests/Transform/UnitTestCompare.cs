// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.IO;
using System.Linq;
using Nncase.IR;
using Nncase.Passes;
using Nncase.Tests.TestFixture;
using Nncase.TIR;
using Xunit;
using static Nncase.IR.F.Math;
using static Nncase.IR.F.Tensors;

namespace Nncase.Tests.TransformTest;

public class UnitTestCompare
{
    [Fact]
    public void TestGetChannelAxis()
    {
        Assert.Equal(1, TensorUtil.GetChannelAxis(new[] { 1, 2, 4, 8 }));
        Assert.Equal(1, TensorUtil.GetChannelAxis(new Shape(new[] { 1, 2, 4, 8 })));
    }

    [Fact]
    public void TestGetShapeInfo()
    {
        var expr = TensorUtil.SliceByChannel(new[] { 1, 3, 16, 16 });
        Assert.Equal(new Tensor[] { 1, 3, 16, 16 }, expr);
    }
}
