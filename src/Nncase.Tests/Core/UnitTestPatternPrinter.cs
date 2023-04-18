// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.IO;
using System.Text;
using Nncase;
using Nncase.PatternMatch;
using Nncase.Tests.TestFixture;
using Xunit;

namespace Nncase.Tests.CoreTest;

public sealed class UnitTestPatternPrinter
{
    [Fact]
    public void TestDumpAsIL()
    {
        IPattern pattern = null!;
        Assert.Throws<NullReferenceException>(() => PatternPrinter.DumpAsIL(pattern));
    }
}
