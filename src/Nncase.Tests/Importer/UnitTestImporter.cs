// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IO;
using Nncase.IR;
using Nncase.Passes.Transforms;
using Nncase.Tests.TestFixture;
using Xunit;

namespace Nncase.Tests.ImporterTest;

[AutoSetupTestMethod(InitSession = true)]
public class UnitTestImporter : TestClassBase
{
    [Fact]
    public async Task TestImportOnnx()
    {
        using var file = File.OpenRead(Path.Join(SolutionDirectory, "examples/user_guide/test.onnx"));
        var module = Importers.ImportOnnx(file, CompileSession);
        await InferShapeAsync(module);
        Assert.NotNull(module.Entry);
        Assert.True(module.Entry!.InferenceType());
    }

    [Fact]
    public async Task TestImportTFLite()
    {
        using var file = File.OpenRead(Path.Combine(SolutionDirectory, "examples/user_guide/test.tflite"));
        var module = Importers.ImportTFLite(file, CompileSession);
        await InferShapeAsync(module);
        Assert.NotNull(module.Entry);
        Assert.True(module.Entry!.InferenceType());
    }

    [Fact]
    public async Task TestImportNcnn()
    {
        using var file = File.OpenRead(Path.Combine(SolutionDirectory, "examples/user_guide/test.param"));
        var module = Importers.ImportNcnn(file, new ZeroStream(), CompileSession);
        await InferShapeAsync(module);
        Assert.NotNull(module.Entry);
        Assert.True(module.Entry!.InferenceType());
    }

    private Task InferShapeAsync(IRModule module)
    {
        var pmgr = CompileSession.CreatePassManager("InferShapePasses");
        pmgr.Add<ShapeInferPass>();
        return pmgr.RunAsync(module);
    }
}
