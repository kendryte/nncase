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

    [Fact]
    public void TestDumpAsILCallPattern()
    {
        var wc1 = IsWildcard();
        var wc2 = IsWildcard();
        var pattern = IsCall(null, new FunctionPattern(wc1 + wc2, IsVArgs(wc1, wc2), null));
        var il = pattern.DumpAsIL();
        var builder = new StringBuilder();
        var writer = new StringWriter(builder);
        PatternPrinter.DumpAsIL(writer, pattern);
        Assert.Equal(builder.ToString(), il);
    }

    [Fact]
    public void TestDumpAsILConstPattern()
    {
        var pattern = IsConst(new TypePattern(new TensorType(DataTypes.Float32, new[] { 1 })));
        var il = pattern.DumpAsIL();
        var builder = new StringBuilder();
        var writer = new StringWriter(builder);
        PatternPrinter.DumpAsIL(writer, pattern);
        Assert.Equal(builder.ToString(), il);
    }

    [Fact]
    public void TestDumpAsILTensorConstPattern()
    {
        var pattern = IsTensorConst(new TypePattern(new TensorType(DataTypes.Float32, new[] { 1 })));
        var il = pattern.DumpAsIL();
        var builder = new StringBuilder();
        var writer = new StringWriter(builder);
        PatternPrinter.DumpAsIL(writer, pattern);
        Assert.Equal(builder.ToString(), il);
    }

    [Fact]
    public void TestDumpAsILTupleConstPattern()
    {
        var pattern = IsTupleConst(new TypePattern(new TensorType(DataTypes.Float32, new[] { 1 })));
        var il = pattern.DumpAsIL();
        var builder = new StringBuilder();
        var writer = new StringWriter(builder);
        PatternPrinter.DumpAsIL(writer, pattern);
        Assert.Equal(builder.ToString(), il);
    }

    [Fact]
    public void TestDumpAsILTuplePattern()
    {
        var pattern = IsTuple(string.Empty);
        var il = pattern.DumpAsIL();
        var builder = new StringBuilder();
        var writer = new StringWriter(builder);
        PatternPrinter.DumpAsIL(writer, pattern);
        Assert.Equal(builder.ToString(), il);
    }

    [Fact]
    public void TestDumpAsILVarPattern()
    {
        var pattern = IsVar(new TypePattern(new TensorType(DataTypes.Float32, new[] { 1 })));
        var il = pattern.DumpAsIL();
        var builder = new StringBuilder();
        var writer = new StringWriter(builder);
        PatternPrinter.DumpAsIL(writer, pattern);
        Assert.Equal(builder.ToString(), il);
    }
}
