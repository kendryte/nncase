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
using Xunit;

namespace Nncase.Tests.CoreTest;

public class UnitTestExpressionOutputsNames
{
    [Fact]
    public void TestExpressionOutputsNames()
    {
        var a = new Var("a", TensorType.Scalar(DataTypes.Float32));
        var meta = a.Metadata;
        Assert.NotNull(meta);
        Assert.Null(meta.OutputsNames);
        meta.OutputsNames = new string[] { "a" };
        Assert.NotNull(meta.OutputsNames);
    }
}
