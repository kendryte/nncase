// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Linq.Expressions;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Options;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.NN;
using Nncase.Utilities;
using Xunit;
using static Nncase.IR.F.NN;

namespace Nncase.Tests.CoreTest;

public class UnitTestExpressionOutputNames
{
    [Fact]
    public void TestExpressionOutputNames()
    {
        var a = new Var("a", TensorType.Scalar(DataTypes.Float32));
        var meta = a.Metadata;
        Assert.NotNull(meta);
        Assert.Null(meta.OutputNames);
        meta.OutputNames = new string[] { "a" };
        Assert.NotNull(meta.OutputNames);
    }

    [Fact]
    public void TestInheritMetaData()
    {
        var input = new Var(new TensorType(DataTypes.Float32, new Shape(1, 3, 224, 224)));
        var pad1 = Pad(input, new float[,] { { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 } }, PadMode.Constant, 0.0f);
        pad1.Metadata.OutputNames = new string[] { "pad" };
        var pad2 = Pad(input, new float[,] { { 0, 0 }, { 1, 1 }, { 1, 1 }, { 0, 0 } }, PadMode.Constant, 0.0f).InheritMetaData(pad1);
        Assert.NotNull(pad2.Metadata.OutputNames);
    }
}
