// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualBasic;
using Nncase;
using Nncase.CodeGen;
using Nncase.IR;
using Nncase.Utilities;
using Xunit;

namespace Nncase.Tests.CoreTest;

public struct Enm
{
    public string Title;
    public string Author;
    public string Subject;
    public int BookId;
}

public class UnitTestCodeGenUtil
{
    [Fact]
    public void TestCodeGenUtil()
    {
        CodeGenUtil.GetTempFileName();
    }
}
