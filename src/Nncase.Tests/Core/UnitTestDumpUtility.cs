// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using Nncase;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.Utilities;
using Xunit;

namespace Nncase.Tests.CoreTest;

public sealed class UnitTestDumpUtility
{
    [Fact]
    public void TestValueDumper()
    {
        ValueDumper.DumpTensor(new TensorValue(new Tensor<int>(new[] { 1 })), "./test1");
        ValueDumper.DumpTensors(new[] { new TensorValue(new Tensor<int>(new[] { 1 })) }, "./test2");
        Assert.True(File.Exists("./test1"));
        Assert.True(Directory.Exists("./test2"));
    }

    [Fact]
    public void TestDumpUtility()
    {
        DumpUtility.WriteResult("./test3", "1");
        DumpUtility.WriteResult<int>("./test3", new[] { 1 });
        DumpUtility.SerializeShape(new[] { 1, 1, 1 });
        DumpUtility.PathJoinByCreate("./", "test4");
        DumpUtility.WriteBinFile("./test5", new Tensor<int>(new[] { 1 }));
        Assert.True(File.Exists("./test3"));
        Assert.True(Directory.Exists("./test4"));
        Assert.True(File.Exists("./test5"));
    }

    [Fact]
    public void TestBinFileUtil()
    {
        BinFileUtil.WriteBinInputs(new Tensor[] { new Tensor<int>(new[] { 1 }) }, "./");
        BinFileUtil.WriteBinOutputs(new Tensor[] { new Tensor<int>(new[] { 1 }) }, "./");
        Assert.True(File.Exists("./input_0_0.bin"));
        Assert.True(File.Exists("./nncase_result_0.bin"));
    }

    [Fact]
    public void TestNullDumpper()
    {
        var nullDumpper = new NullDumpper();
        string nullDumpperDirectory = nullDumpper.Directory;
        nullDumpper.DumpIR(1, string.Empty);
        nullDumpper.DumpDotIR(1, string.Empty);
        nullDumpper.DumpModule(new IRModule(), string.Empty);
        nullDumpper.DumpCSharpIR(1, string.Empty);
        nullDumpper.OpenFile("./");
    }
}
