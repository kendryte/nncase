// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices.JavaScript;
using NetFabric.Hyperlinq;
using Nncase;
using Nncase.IR;
using Nncase.IR.Random;
using Nncase.Tests.TestFixture;
using Xunit;

namespace Nncase.Tests.CoreTest;

public sealed class UnitTestFusion
{
    [Fact]
    public void FusionWith_UpdatesProperties()
    {
        var fusion = new Fusion("myFunc", "module", 1, new Var("x", DataTypes.Int32));

        var updatedFusion = fusion.With(name: "newFunc", body: default(bool), parameters: new[] { new Var("y", DataTypes.Boolean) });

        Assert.Equal("newFunc", updatedFusion.Name);
        Assert.Equal(DataTypes.Boolean, updatedFusion.Body.CheckedType);
        Assert.Equal(1, updatedFusion.Parameters.Length);
        Assert.Equal("y", updatedFusion.Parameters[0].Name);
        Assert.Equal(DataTypes.Boolean, updatedFusion.Parameters[0].CheckedType);
        Assert.Equal("module", updatedFusion.ModuleKind);

        var readOnlySpan = fusion.Parameters;
        var expect = readOnlySpan.AsValueEnumerable().Select(x => x.CheckedType).ToArray();
        Assert.Equal(expect, fusion.ParameterTypes);

        Assert.NotNull(new Fusion("module", 1, default(ReadOnlySpan<Var>)));
    }
}
