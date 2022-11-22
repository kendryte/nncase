// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.Transform;
using Nncase.Transform.Passes;
using Xunit;

namespace Nncase.Tests.ImporterTest;

public class UnitTestImporter : TestFixture.UnitTestFixtrue
{
    [Fact]
    public async Task TestImport20ClassesYolo()
    {
        using var file = File.OpenRead(Path.Combine(GetSolutionDirectory(), "examples/20classes_yolo/model/20classes_yolo.tflite"));
        var options = GetCompileOptions();
        var module = Importers.ImportTFLite(file, options);
        await InferShape(module, options);
        Assert.True(module.Entry!.InferenceType());
    }

    private async Task InferShape(IRModule module, CompileOptions options)
    {
        var pmgr = new PassManager(module, new RunPassOptions(null!, options.DumpLevel, options.DumpDir));
        var constFold = new ShapeInferPass();
        pmgr.Add(constFold);
        await pmgr.RunAsync();
    }
}
