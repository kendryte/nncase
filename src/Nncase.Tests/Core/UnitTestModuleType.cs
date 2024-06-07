// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Nncase;
using Nncase.CodeGen;
using Nncase.IR;
using Xunit;

namespace Nncase.Tests.CoreTest;

public sealed class UnitTestModuleType
{
    [Fact]
    public void TestModuleType()
    {
        var expect = default(ModuleType);
        expect.Types = "test" + "\0\0\0\0\0\0\0\0\0\0\0\0"; // type.length()==16.
        Assert.Equal(expect, ModuleType.Create("test"));
    }
}
