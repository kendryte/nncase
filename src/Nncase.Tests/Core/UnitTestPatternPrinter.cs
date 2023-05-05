// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Microsoft.Extensions.Hosting;
using Nncase.IR;
using Nncase.IR.NN;
using Nncase.Passes;
using Nncase.PatternMatch;
using Nncase.PatternMatch.F;
using Nncase.TIR;
using Tensorflow;
using Xunit;
using static Nncase.IR.F.Math;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;
using Function = Nncase.IR.Function;
using Math = Nncase.PatternMatch.F.Math;

namespace Nncase.Tests.CoreTest;

public sealed class UnitTestPatternPrinter
{
    [Fact]
    public void TestDumpAsIL()
    {
        var pattern = IsNone();
        var s = PatternPrinter.DumpAsIL(pattern);
        Assert.Equal(s, string.Empty);
        PatternPrinter.DumpAsIL(pattern, "dumpAsIl", "./");
        var builder = new StringBuilder();
        var writer = new StringWriter(builder);
        PatternPrinter.DumpAsIL(writer, pattern);
        Assert.Equal(builder.ToString(), string.Empty);
    }
}
