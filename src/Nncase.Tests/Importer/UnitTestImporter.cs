// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.Passes.Transforms;
using Nncase.Tests.TestFixture;
using Xunit;

namespace Nncase.Tests.ImporterTest;

[AutoSetupTestMethod(InitSession = true)]
public class UnitTestImporter : TestClassBase
{
    [Fact]
    public async Task TestImport20ClassesYolo()
    {
        using var file = File.OpenRead(Path.Join(SolutionDirectory, "examples/20classes_yolo/model/20classes_yolo.tflite"));
        var module = Importers.ImportTFLite(file, CompileSession);
        await InferShapeAsync(module);
        Assert.NotNull(module.Entry);
        Assert.True(module.Entry!.InferenceType());
    }

    [Fact]
    public async Task TestImportFacedetectLandmark()
    {
        using var file = File.OpenRead(Path.Combine(SolutionDirectory, "examples/facedetect_landmark/model/ulffd_landmark.tflite"));
        var module = Importers.ImportTFLite(file, CompileSession);
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
