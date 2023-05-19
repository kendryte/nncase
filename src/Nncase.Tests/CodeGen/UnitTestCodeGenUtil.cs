// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.VisualBasic;
using Nncase;
using Nncase.CodeGen;
using Nncase.CodeGen.StackVM;
using Nncase.IR;
using Nncase.Utilities;
using Xunit;

namespace Nncase.Tests.CoreTest;

public class UnitTestCodeGenUtil
{
    [Fact]
    public void TestCodeGenUtil()
    {
        string tempPath = Path.GetTempPath() + Guid.NewGuid().ToString();
        Assert.NotEqual(tempPath, CodeGenUtil.GetTempFileName());
    }

    [Fact]
    public void TestStructToBytes()
    {
        var num = new[] { new byte[] { 1, 2, 3 }, new byte[] { 2, 3, 4 }, new byte[] { 3, 4, 5 } };
        Assert.Throws<ArgumentException>(() => CodeGenUtil.StructToBytes(num));
    }

    [Fact]
    public void TestStackVMModuleBuilder()
    {
        string moduleKind = new StackVMModuleBuilder().ModuleKind;
        Assert.Equal("stackvm", moduleKind);
    }
}
